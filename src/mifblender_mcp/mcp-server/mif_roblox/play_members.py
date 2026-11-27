"""rbx_play_probe kind "members" (spec: "Extend rbx_play_probe: kind members, typed reads, calls and watches inside the play
test" and "cause steps, so events fire"): the MCP half.

A plan is a list of typed steps. Each is checked against the catalog here, before any session starts, and sent as data:
the MifBridge plugin that Studio loads into the play server (and into each play client) runs it with its own resolve,
decode and encode - the same code that answers in Edit - so no caller Luau crosses.

    {"read": ["UserId", "DisplayName"], "target": "@player"}            properties of a target
    {"call": "GetState", "args": [], "target": "@character.Humanoid", "class": "Humanoid"}   a read function
    {"watch": "HealthChanged", "target": "@character.Humanoid", "class": "Humanoid"}         an event, from now to the end
    {"cause": "damage", "amount": 10}                                    something that makes an event fire (below)
    {"wait": 1.5}                                                        seconds before the next step

Every step has "on": "server" (default) or "client" (the first player's client). Targets: a path from game that exists in
Edit (its class read from Studio now), or a play root - @player (Players:GetPlayers()[1] on the server, LocalPlayer on a
client), @character, @playergui, @camera (Workspace.CurrentCamera) and @mouse (client only) - optionally followed by
child names, then "class" names what the step expects there and the plugin checks IsA in play.

Causes (S14), each named in the answer beside the watched events that fired in its window, each with firedBefore;
a cause may carry expect, the "Class.Event" names of this plan's watches it should raise, which decides raisedNothing:
    key {key, seconds}      client  UserInputService:CreateVirtualInput():SendKey down then up (RB1440: arrives as Keyboard)
    fire {target, value}    either  InputBinding:Fire(value) on a Scriptable binding (then Fire(false) after true)
    hold {target}           either  ProximityPrompt:InputHoldBegin, its HoldDuration, InputHoldEnd
    move_to {point}         server  the character's Humanoid:MoveTo(point)
    change_state {state}    either  Humanoid:ChangeState(state) (a player's character is its client's: state changes act there)
    damage {amount}         server  Humanoid:TakeDamage(amount)
    equip {}                server  a scratch Tool with a Handle into the Backpack, then Humanoid:EquipTool
    touch {}                server  a scratch anchored part moved onto the character's HumanoidRootPart
    remote {args, unreliable}  either  a client fires ReplicatedStorage.MifMembersRemote (made for the run) to the server
                            with args; on the server, FireClient to the first player (plugin 0.17.19); unreliable true uses
                            MifMembersUnreliable, an UnreliableRemoteEvent made for the run
    leave {}                client  StudioTestService:LeaveTest() - the server sees PlayerRemoving; that client reports nothing
  plugin 0.17.19 (pointer input from a play client: VirtualInput's mouse members work there, RB2740):
    mouse_button {button, at|position, seconds, drag}  client  SendMouseButton down, held seconds (default 0.1), up; drag
                            [dx, dy] moves the pointer that far in steps while held and releases there
    mouse_move {at|position}  client  SendMousePosition
    mouse_delta {delta}     client  SendMouseDelta
    pointer_action {at|position, action}  client  SendPointerAction(position, action): action is a small table of named
                            numbers or [x, y] pairs, passed as it is given
    text_input {text}       client  SendTextInput (a TextBox must have focus: the focus cause)
    focus {target}          client  TextBox:CaptureFocus on a TextBox
    set {target, props}     either  (plugin 0.17.20) properties written in play and read back (a camera made
                            Scriptable, UserInputService.MouseBehavior LockCenter for mouse_delta: "cursor is not locked")
    make {class, parent, name, props, near}  either  Instance.new(class) with props, under parent (a play root path, a path
                            that exists in Edit, or a runtime path); near [x, y, z] puts a part at that offset in the
                            character's HumanoidRootPart frame. Destroyed at the end of the window
    unequip {}              server  the character's Humanoid:UnequipTools
    call {target, method, args, check}  either  one member of PLAY_CALLS (an opt-in list, never a blind call) with typed
                            args; check {prop} or {call, args} on target (or check.target) is read before and after
  at is a target (a GuiObject: its centre plus the GUI inset, RB2740; a BasePart: its WorldToViewportPoint), position is
  [x, y] in the same screen space.

A step may give "runtime": true with class: its target is a path from a service that only exists in play (made by a make
cause, or by Roblox: TextChatService.TextChannels), not resolved in Edit; the plugin waits for it in play and checks IsA.
An argument of a Class.* parameter may name a play root (@player ...): the plugin resolves it in play.

What a step proved goes to the coverage record (spec "members proven inside a play test count"): a property read whose
value decoded is play-read, a read call that answered is play-call, a watch that saw a firing is play-event - only from
a result whose nonce is this run's and that came back as the reply, never one recovered from a log. A cause's own members
(CAUSE_PROVES; PLAY_CALLS for a call) are play-call-verified when the cause ran and one of the events that prove it
fired in its window and not in the ALREADY_MS before it, or when its check read back changed.
"""
from mif_roblox import catalog as C
from mif_roblox import values as V

ROOTS = {"@player": "Player", "@character": "Model", "@playergui": "PlayerGui", "@camera": "Camera", "@mouse": "PlayerMouse",
         "@input": "InputObject", "@track": "AnimationTrack"}
# @input (plugin 0.17.19): the last InputObject a watch on that client was passed; @track: the last animate cause's
# AnimationTrack on that side
CLIENT_ONLY_ROOTS = {"@mouse", "@input"}
CAUSES = {"key": ("client",), "fire": ("server", "client"), "hold": ("server", "client"), "move_to": ("server",),
          "change_state": ("server", "client"), "damage": ("server",), "equip": ("server",), "touch": ("server",),
          "remote": ("client", "server"), "leave": ("client",),
          # plugin 0.17.19
          "mouse_button": ("client",), "mouse_move": ("client",), "mouse_delta": ("client",),
          "pointer_action": ("client",), "text_input": ("client",), "focus": ("client",), "make": ("server", "client"),
          "unequip": ("server",), "call": ("server", "client"),
          "activate": ("server", "client"), "deactivate": ("server", "client"), "chat": ("client",),
          "system_message": ("client",), "animate": ("server", "client"), "set": ("server", "client")}
CAUSES_PLUGIN = {c: (0, 17, 19) for c in ("mouse_button", "mouse_move", "mouse_delta", "pointer_action", "text_input",
                                          "focus", "make", "unequip", "call", "activate", "deactivate", "chat",
                                          "system_message", "animate")}
CAUSES_PLUGIN["set"] = (0, 17, 20)
MAX_STEPS = 2000
MAX_UNITS_PER_RUN = 400        # props + calls + watches + causes in one play session; a larger plan runs in parts
REMOTE_NAME = "MifMembersRemote"
UNRELIABLE_NAME = "MifMembersUnreliable"
SIDES = ("server", "client")
MOUSE_BUTTONS = ("MouseButton1", "MouseButton2", "MouseButton3")

# What each cause's own members are proven by: an event (Owner.Event, keyed by the class that declares it) that fired in
# the cause's window and not in the ALREADY_MS before it. The plugin calls exactly these members for the cause.
_KEY = ("UserInputService.InputBegan", "UserInputService.InputEnded")
_MOUSE = ("UserInputService.InputBegan", "UserInputService.InputEnded", "GuiButton.MouseButton1Down",
          "GuiButton.MouseButton1Up", "GuiButton.MouseButton2Down", "GuiButton.MouseButton2Up", "Mouse.Button1Down",
          "Mouse.Button1Up", "Mouse.Button2Down", "Mouse.Button2Up", "GuiObject.InputBegan", "GuiObject.InputEnded",
          "ClickDetector.MouseClick", "ClickDetector.RightMouseClick")
_MOVE = ("UserInputService.InputChanged", "Mouse.Move", "GuiObject.MouseEnter", "GuiObject.MouseLeave",
         "GuiObject.MouseMoved", "GuiObject.InputChanged")
_POINTER = ("UserInputService.PointerAction", "Mouse.WheelForward", "Mouse.WheelBackward", "GuiObject.MouseWheelForward",
            "GuiObject.MouseWheelBackward")
CAUSE_PROVES = {
    "key": {"VirtualInput.SendKey": _KEY, "UserInputService.CreateVirtualInput": _KEY},
    "mouse_button": {"VirtualInput.SendMouseButton": _MOUSE, "UserInputService.CreateVirtualInput": _MOUSE},
    "mouse_move": {"VirtualInput.SendMousePosition": _MOVE},
    "mouse_delta": {"VirtualInput.SendMouseDelta": _MOVE},
    "pointer_action": {"VirtualInput.SendPointerAction": _POINTER},
    "focus": {"TextBox.CaptureFocus": ("TextBox.Focused", "UserInputService.TextBoxFocused")},
    "hold": {"ProximityPrompt.InputHoldBegin": ("ProximityPrompt.PromptButtonHoldBegan", "ProximityPrompt.Triggered",
                                                "ProximityPromptService.PromptButtonHoldBegan",
                                                "ProximityPromptService.PromptTriggered"),
             "ProximityPrompt.InputHoldEnd": ("ProximityPrompt.PromptButtonHoldEnded", "ProximityPrompt.TriggerEnded",
                                              "ProximityPromptService.PromptButtonHoldEnded",
                                              "ProximityPromptService.PromptTriggerEnded")},
    "fire": {"InputBinding.Fire": ("InputAction.Pressed", "InputAction.Released", "InputAction.StateChanged")},
    "damage": {"Humanoid.TakeDamage": ("Humanoid.HealthChanged",)},
    "move_to": {"Humanoid.MoveTo": ("Humanoid.MoveToFinished",)},
    "change_state": {"Humanoid.ChangeState": ("Humanoid.StateChanged",)},
    "equip": {"Humanoid.EquipTool": ("Tool.Equipped", "ContextActionService.LocalToolEquipped")},
    "unequip": {"Humanoid.UnequipTools": ("Tool.Unequipped", "ContextActionService.LocalToolUnequipped")},
    "leave": {"StudioTestService.LeaveTest": ("Players.PlayerRemoving",)},
    "activate": {"Tool.Activate": ("Tool.Activated",)},
    "deactivate": {"Tool.Deactivate": ("Tool.Deactivated",)},
    "chat": {"TextChannel.SendAsync": ("TextChannel.MessageReceived", "TextChatService.MessageReceived",
                                       "TextChatService.SendingMessage")},
    "system_message": {"TextChannel.DisplaySystemMessage": ("TextChannel.MessageReceived",
                                                            "TextChatService.MessageReceived")},
}
# causes whose plugin reads a check of its own (detail.check before and after): the members a change proves
CHECK_PROVES = {"text_input": ("VirtualInput.SendTextInput",),
                "animate": ("Animator.LoadAnimation", "AnimationTrack.Play")}
# remote: which member fired depends on the side and unreliable (judge picks the row)
REMOTE_PROVES = {("client", False): {"RemoteEvent.FireServer": ("RemoteEvent.OnServerEvent",)},
                 ("client", True): {"UnreliableRemoteEvent.FireServer": ("UnreliableRemoteEvent.OnServerEvent",)},
                 ("server", False): {"RemoteEvent.FireClient": ("RemoteEvent.OnClientEvent",)},
                 ("server", True): {"UnreliableRemoteEvent.FireClient": ("UnreliableRemoteEvent.OnClientEvent",)}}

# The call cause's opt-in list (plan section 2.5: functions are rows, never swept blind). Each row: the sides it runs on
# and what proves it - events (as in CAUSE_PROVES) and/or "check" (a read before and after that must differ). A row is
# kept only after a live run showed its proof holds. Measured and left out: TestService:Check
# and Message from a client raise no ServerCollectResult on the server.
PLAY_CALLS = {
    "GuiService.Select": {"on": ("client",), "events": ("GuiObject.SelectionGained", "GuiBase2d.SelectionChanged"),
                          "check": True},
    "TextBox.ReleaseFocus": {"on": ("client",), "events": ("TextBox.FocusLost", "UserInputService.TextBoxFocusReleased")},
    "UIPageLayout.Next": {"on": ("client",), "events": ("UIPageLayout.PageEnter", "UIPageLayout.PageLeave",
                                                         "UIPageLayout.Stopped")},
    "Camera.SetRoll": {"on": ("client",), "events": (), "check": True},     # on a Scriptable camera (Custom reads 0)
    "BasePart.ApplyImpulse": {"on": ("server",), "events": (), "check": True},
    "BasePart.ApplyImpulseAtPosition": {"on": ("server",), "events": (), "check": True},
    "BasePart.ApplyAngularImpulse": {"on": ("server",), "events": (), "check": True},
    "BasePart.SetNetworkOwner": {"on": ("server",), "events": (), "check": True},
    "RemoteEvent.FireAllClients": {"on": ("server",), "events": ("RemoteEvent.OnClientEvent",)},
    "UnreliableRemoteEvent.FireAllClients": {"on": ("server",), "events": ("UnreliableRemoteEvent.OnClientEvent",)},
    "ScriptProfilerService.ClientStart": {"on": ("server",), "events": ()},
    "ScriptProfilerService.ClientRequestData": {"on": ("server",), "events": ("ScriptProfilerService.OnNewData",),
                                                "requires": ("ScriptProfilerService.ClientStart",)},
    "HeapProfilerService.ClientRequestDataAsync": {"on": ("server",), "events": ("HeapProfilerService.OnNewData",)},
    # on @track, after an animate cause (CV3-B's animation members; kept once CV3-B's live run shows each proof holds)
    "AnimationTrack.Stop": {"on": ("server", "client"), "events": ("AnimationTrack.Stopped", "AnimationTrack.Ended")},
    "AnimationTrack.AdjustSpeed": {"on": ("server", "client"), "events": (), "check": True},
    "AnimationTrack.AdjustWeight": {"on": ("server", "client"), "events": (), "check": True},
}
CHECK_WAIT = 0.3      # seconds the plugin waits after a call before its check's second read (a physics step, a render)


def _bad(i, why, **extra):
    return dict({"ok": False, "ran": False, "refusedRule": extra.pop("rule", "plan"), "step": i,
                 "error": "plan step %d: %s. Nothing was started" % (i, why)}, **extra)


def _target(step, i, side, resolve):
    """(parts, class, None) or (None, None, refusal) for a step's target."""
    t = step.get("target")
    parts = V.split_path(t)
    if not parts:
        return None, None, _bad(i, "target is a path from game or a play root (%s), got %r" % (", ".join(ROOTS), t))
    root = parts[0]
    if parts == ["ReplicatedStorage", REMOTE_NAME]:
        return parts, "RemoteEvent", None      # made for the run when a client step causes remote; not in Edit before
    if parts == ["ReplicatedStorage", UNRELIABLE_NAME]:
        return parts, "UnreliableRemoteEvent", None
    if step.get("runtime") is not None and not root.startswith("@"):
        # a path that exists only in play (a make cause's object, Roblox's own TextChannels): not resolved in Edit, waited
        # for in play, its class checked there by IsA
        cls = step.get("class")
        if step.get("runtime") is not True or not isinstance(cls, str):
            return None, None, _bad(i, "runtime is true, with class: what the step expects at %s in play" % V.join_path(parts))
        return parts, cls, None
    if root.startswith("@"):
        if root not in ROOTS:
            return None, None, _bad(i, "%s is not a play root; the roots are %s" % (root, ", ".join(ROOTS)))
        if root in CLIENT_ONLY_ROOTS and side != "client":
            return None, None, _bad(i, "%s exists only on a client: give the step on client" % root)
        cls = step.get("class")
        if len(parts) > 1 and not isinstance(cls, str):
            return None, None, _bad(i, "a target below %s (%s) needs class: what the step expects there" % (
                root, V.join_path(parts)))
        return parts, cls or ROOTS[root], None
    cls, err = resolve(parts)
    if err:
        return None, None, _bad(i, "target %s does not resolve in Edit: %s" % (V.join_path(parts), err.get("error")),
                                rule=err.get("refusedRule") or "not-found")
    want = step.get("class")
    if isinstance(want, str) and want != cls:
        return None, None, _bad(i, "target %s is a %s, not the %s the step names" % (V.join_path(parts), cls, want),
                                rule="wrong-class")
    return parts, cls, None


def check_plan(cat, plan, resolve, read_only):
    """(wire steps, None) or (None, the refusal). resolve(parts) -> (className, err) reads a path's class in Edit;
    read_only(name, owner) is tools' read-by-name judgement. Nothing is sent to Studio from here but resolve."""
    if not isinstance(plan, list) or not plan:
        return None, {"ok": False, "ran": False, "refusedRule": "plan", "error": "members needs plan: a list of steps"}
    if len(plan) > MAX_STEPS:
        return None, {"ok": False, "ran": False, "refusedRule": "plan", "error": "a plan has at most %d steps" % MAX_STEPS}
    out = []
    for i, step in enumerate(plan):
        if not isinstance(step, dict):
            return None, _bad(i, "a step is an object")
        side = step.get("on", "server")
        if side not in SIDES:
            return None, _bad(i, "on is server or client")
        verbs = [k for k in ("read", "call", "watch", "cause", "wait") if k in step]
        if len(verbs) != 1:
            return None, _bad(i, "a step has exactly one of read, call, watch, cause or wait")
        verb = verbs[0]
        w = {"i": i, "on": side, "op": verb}
        if verb == "wait":
            s = step["wait"]
            if not isinstance(s, (int, float)) or isinstance(s, bool) or not 0 <= s <= 60:
                return None, _bad(i, "wait is 0-60 seconds")
            w["seconds"] = s
        elif verb == "cause":
            w, bad = _cause(cat, step, i, side, resolve, w, read_only)
            if bad:
                return None, bad
        else:
            parts, cls, bad = _target(step, i, side, resolve)
            if bad:
                return None, bad
            if cls not in cat.classes:
                return None, dict(cat.unknown_class(cls), step=i)
            w.update(target=parts, **{"class": cls})
            if step.get("runtime") is True:
                w["runtime"] = True
            if verb == "read":
                props = step["read"]
                if not isinstance(props, list) or not props or not all(isinstance(p, str) for p in props):
                    return None, _bad(i, "read is a list of property names")
                for p in props:
                    bad = cat.read_refusal(cls, p, kind="P")
                    if bad:
                        return None, dict(bad, step=i)
                w["props"] = list(props)
            elif verb == "watch":
                ev = step["watch"]
                bad = cat.read_refusal(cls, ev, kind="E") if isinstance(ev, str) else _bad(i, "watch is an event name")
                if bad:
                    return None, dict(bad, step=i)
                _o, row = cat.find_member(cls, ev)
                if any(p[1] == "Function" for p in row[5]):
                    return None, _bad(i, "%s passes a function" % ev, rule="needs-function")
                w["event"] = ev
                if step.get("early") is not None:
                    # plugin 0.17.19: armed when the plugin loads in the play server, before any player joins
                    if step["early"] is not True or side != "server" or parts[0].startswith("@"):
                        return None, _bad(i, "early is true, on a server watch of a path in the place (no play root: "
                                             "nothing of a player exists before one joins)")
                    w["early"] = True
            else:
                m = step["call"]
                if not isinstance(m, str) or not m:
                    return None, _bad(i, "call is a function name")
                bad = cat.read_refusal(cls, m, kind="F")
                if bad:
                    return None, dict(bad, step=i)
                owner, row = cat.find_member(cls, m)
                if not read_only(m, owner):
                    return None, _bad(i, "%s:%s is not a read by name; in play a plan calls reads only - what changes "
                                         "the game is a cause step" % (owner, m), rule="play-write")
                params = row[5]
                if any(p[1] == "Function" for p in params):
                    return None, _bad(i, "%s:%s takes a Luau function" % (owner, m), rule="needs-function")
                args = step.get("args", [])
                if not isinstance(args, list) or len(args) > len(params):
                    return None, _bad(i, "%s:%s takes %d arguments" % (owner, m, len(params)), rule="type")
                for p in params[len(args):]:
                    if p[2] is None and not str(p[1]).endswith("?"):
                        return None, _bad(i, "%s:%s needs %s (%s)" % (owner, m, p[0], p[1]), rule="type")
                wire, bad = _encode_args(cat, params, args, i, side)
                if bad:
                    return None, bad
                w.update(method=m, args=wire, argc=len(wire))
        out.append(w)
    # expect on a cause: the "Class.Event" names, of this plan's watches, that the cause should raise
    watched = {"%s.%s" % (s["class"], s["event"]) for s in out if s["op"] == "watch"}
    for s, step in zip(out, plan):
        if s["op"] != "cause" or "expect" not in step:
            continue
        e = step["expect"]
        if not isinstance(e, list) or not e or not all(isinstance(x, str) for x in e):
            return None, _bad(s["i"], "expect is a list of Class.Event names of this plan's watches")
        stray = [x for x in e if x not in watched]
        if stray:
            return None, _bad(s["i"], "expect names %s, which no watch step watches (watched: %s)" % (
                stray, ", ".join(sorted(watched)) or "none"))
        s["expect"] = list(e)
    return out, None


def _play_arg(value, side):
    """The parts of a play-root path given as an argument ('@player', '@playergui.Gui'), or None when it is not one."""
    parts = V.split_path(value) if isinstance(value, (str, list)) else None
    if not parts or not str(parts[0]).startswith("@"):
        return None
    return parts


def _encode_args(cat, params, args, i, side):
    """(wire args, None) or (None, refusal): each by its catalog type; a Class.* parameter may take a play root path."""
    wire = []
    for p, value in zip(params, args):
        parts = _play_arg(value, side) if str(p[1]).rstrip("?").startswith("Class.") else None
        if parts is not None:
            if parts[0] not in ROOTS:
                return None, _bad(i, "%s: %s is not a play root; the roots are %s" % (p[0], parts[0], ", ".join(ROOTS)),
                                  rule="type")
            if parts[0] in CLIENT_ONLY_ROOTS and side != "client":
                return None, _bad(i, "%s: %s exists only on a client" % (p[0], parts[0]), rule="type")
            wire.append({"$t": "Instance", "path": parts, "play": True})
            continue
        x, why = V.encode(p[1], value, cat.enums)
        if why:
            return None, _bad(i, "%s: %s" % (p[0], why), rule="type")
        wire.append(x)
    return wire, None


def _point(cat, step, i, resolve, w, need=True):
    """at (a GuiObject or BasePart target) or position [x, y] into w; a refusal or None."""
    at, pos = step.get("at"), step.get("position")
    if at is not None and pos is not None:
        return _bad(i, "give at or position, not both")
    if at is not None:
        parts = V.split_path(at)
        if not parts:
            return _bad(i, "at is a target path (a GuiObject or a BasePart)")
        if parts[0].startswith("@"):
            if parts[0] not in ROOTS:
                return _bad(i, "%s is not a play root; the roots are %s" % (parts[0], ", ".join(ROOTS)))
        elif step.get("runtime") is not True:
            cls, err = resolve(parts)
            if err:
                return _bad(i, "at %s does not resolve in Edit: %s (give runtime true for a path made in play)" % (
                    V.join_path(parts), err.get("error")), rule=err.get("refusedRule") or "not-found")
            if not (cat.is_a(cls, "GuiObject") or cat.is_a(cls, "BasePart")):
                return _bad(i, "at %s is a %s; a pointer goes to a GuiObject or a BasePart" % (V.join_path(parts), cls),
                            rule="wrong-class")
        w["at"] = parts
        off = step.get("offset", [0, 0])
        p, why = V.encode("Vector2", off, cat.enums)
        if why:
            return _bad(i, "offset: %s" % why, rule="type")
        w["offset"] = p
        return None
    if pos is not None:
        p, why = V.encode("Vector2", pos, cat.enums)
        if why:
            return _bad(i, "position: %s" % why, rule="type")
        w["position"] = p
        return None
    return _bad(i, "%s needs at (a target) or position [x, y]" % step["cause"]) if need else None


def _cause(cat, step, i, side, resolve, w, read_only=lambda _n, _o=None: False):
    c = step["cause"]
    if c not in CAUSES:
        return None, _bad(i, "cause is one of %s" % ", ".join(sorted(CAUSES)))
    if side not in CAUSES[c]:
        return None, _bad(i, "cause %s acts on the %s; give on %s" % (c, " or ".join(CAUSES[c]), CAUSES[c][0]))
    w["cause"] = c
    if c in ("mouse_button", "mouse_move", "pointer_action"):
        bad = _point(cat, step, i, resolve, w)
        if bad:
            return None, bad
    if c == "mouse_button":
        b, secs = step.get("button", "MouseButton1"), step.get("seconds", 0.1)
        if b not in MOUSE_BUTTONS:
            return None, _bad(i, "button is one of %s" % ", ".join(MOUSE_BUTTONS), rule="type")
        if not isinstance(secs, (int, float)) or isinstance(secs, bool) or not 0 <= secs <= 5:
            return None, _bad(i, "seconds (how long the button is held) is 0-5")
        w.update(button=b, seconds=secs)
        if step.get("drag") is not None:
            d, why = V.encode("Vector2", step["drag"], cat.enums)
            if why:
                return None, _bad(i, "drag: %s" % why, rule="type")
            w["drag"] = d
        return w, None
    if c == "mouse_delta":
        d, why = V.encode("Vector2", step.get("delta"), cat.enums)
        if why:
            return None, _bad(i, "mouse_delta delta: %s" % why, rule="type")
        w["delta"] = d
        return w, None
    if c == "pointer_action":
        a = step.get("action")
        if not isinstance(a, dict) or not a or len(a) > 4:
            return None, _bad(i, "action is a table of up to 4 named values (numbers or [x, y])")
        act = {}
        for k, v in a.items():
            if not isinstance(k, str) or not k.isalpha() or len(k) > 16:
                return None, _bad(i, "an action key is a word, got %r" % (k,))
            if isinstance(v, list):
                x, why = V.encode("Vector2", v, cat.enums)
                if why:
                    return None, _bad(i, "action %s: %s" % (k, why), rule="type")
                act[k] = x
            elif isinstance(v, (int, float)) and not isinstance(v, bool) and V.nonfinite(v) is None:
                act[k] = v
            else:
                return None, _bad(i, "action %s is a number or [x, y], got %r" % (k, v), rule="type")
        w["action"] = act
        return w, None
    if c == "text_input":
        t = step.get("text")
        if not isinstance(t, str) or not t or len(t) > 200:
            return None, _bad(i, "text_input text is 1-200 characters")
        w["text"] = t
        return w, None
    if c == "focus":
        parts = V.split_path(step.get("target"))
        if not parts:
            return None, _bad(i, "focus needs target: a TextBox (a play root path, or runtime true)")
        if not parts[0].startswith("@") and step.get("runtime") is not True:
            cls, err = resolve(parts)
            if err or not cat.is_a(cls, "TextBox"):
                return None, _bad(i, "focus needs a TextBox; %s is %s" % (V.join_path(parts), cls or (err or {}).get("error")),
                                  rule="wrong-class")
        w["target"] = parts
        return w, None
    if c == "make":
        return _make(cat, step, i, resolve, w)
    if c == "set":
        return _set(cat, step, i, side, resolve, w)
    if c == "call":
        return _call_cause(cat, step, i, side, resolve, w, read_only)
    if c == "remote":
        args = step.get("args", [])
        if not isinstance(args, list) or len(args) > 8 or not all(
                isinstance(x, (str, int, float, bool)) for x in args):
            return None, _bad(i, "remote args is a list of up to 8 plain values (text, numbers, true/false)")
        u = step.get("unreliable", False)
        if not isinstance(u, bool):
            return None, _bad(i, "unreliable is true or false")
        w["args"] = list(args)
        if u:
            w["unreliable"] = True
        return w, None
    if c in ("chat", "system_message"):
        t, ch = step.get("text"), step.get("channel", "RBXGeneral")
        if not isinstance(t, str) or not t or len(t) > 200:
            return None, _bad(i, "%s text is 1-200 characters" % c)
        if not isinstance(ch, str) or not ch.replace("_", "").isalnum() or len(ch) > 64:
            return None, _bad(i, "channel is a TextChannel's name (RBXGeneral when left out)")
        w.update(text=t, channel=ch)
        return w, None
    if c in ("equip", "animate"):
        want = "Tool" if c == "equip" else "Animation"
        if step.get("target") is None:
            if c == "animate":
                return None, _bad(i, "animate needs target: an Animation in the place (or made in play, runtime true)")
            return w, None
        parts = V.split_path(step.get("target"))
        if not parts or parts[0].startswith("@"):
            return None, _bad(i, "%s target is the path of a %s" % (c, want))
        if step.get("runtime") is True:
            w["runtime"] = True
        else:
            cls, err = resolve(parts)
            if err or not cat.is_a(cls, want):
                return None, _bad(i, "%s needs a %s; %s is %s" % (c, want, V.join_path(parts),
                                                                 cls or (err or {}).get("error")), rule="wrong-class")
        w["target"] = parts
        if c == "animate" and step.get("looped") is not None:
            if not isinstance(step["looped"], bool):
                return None, _bad(i, "looped is true or false")
            w["looped"] = step["looped"]
        return w, None
    if c in ("activate", "deactivate", "unequip"):
        return w, None
    if c == "key":
        key, secs = step.get("key"), step.get("seconds", 0.1)
        if key not in (cat.enums.get("KeyCode") or []):
            return None, _bad(i, "key is an Enum.KeyCode item name, got %r" % (key,), rule="type")
        if not isinstance(secs, (int, float)) or isinstance(secs, bool) or not 0 <= secs <= 5:
            return None, _bad(i, "seconds (how long the key is held) is 0-5")
        w.update(key=key, seconds=secs)
    elif c in ("fire", "hold"):
        want = "InputBinding" if c == "fire" else "ProximityPrompt"
        parts = V.split_path(step.get("target"))
        if not parts or parts[0].startswith("@"):
            return None, _bad(i, "%s needs target: the path of a %s in the place" % (c, want))
        if step.get("runtime") is True:
            w["runtime"] = True            # made in play (a make cause): the plugin checks the class there
        else:
            cls, err = resolve(parts)
            if err or cls != want:
                return None, _bad(i, "%s needs a %s; %s is %s" % (c, want, V.join_path(parts),
                                                                 cls or (err or {}).get("error")), rule="wrong-class")
        w["target"] = parts
        if c == "fire":
            v = step.get("value")
            if not isinstance(v, (bool, int, float)):
                return None, _bad(i, "fire needs value: true/false, or a number for an analog binding")
            w["value"] = v
    elif c == "move_to":
        if (step.get("point") is None) == (step.get("near") is None):
            return None, _bad(i, "move_to needs point (in the world) or near (an offset in the character's frame), one")
        key = "point" if step.get("point") is not None else "near"
        p, why = V.encode("Vector3", step.get(key), cat.enums)
        if why:
            return None, _bad(i, "move_to %s: %s" % (key, why), rule="type")
        w[key] = p
    elif c == "change_state":
        s = step.get("state")
        if s not in (cat.enums.get("HumanoidStateType") or []):
            return None, _bad(i, "state is an Enum.HumanoidStateType item name, got %r" % (s,), rule="type")
        w["state"] = s
    elif c == "damage":
        a = step.get("amount")
        if not isinstance(a, (int, float)) or isinstance(a, bool) or not 0 < a <= 100000:
            return None, _bad(i, "damage amount is above 0 and at most 100000")
        w["amount"] = a
    return w, None


MAKE_MAX_PROPS = 24


def _encode_props(cat, cls, props, i):
    """({prop: wire}, None) or (None, refusal): each writable on cls by the catalog and typed by it; an object property
    takes a play root path, a path found in play when the cause runs, or null."""
    if not isinstance(props, dict) or len(props) > MAKE_MAX_PROPS:
        return None, _bad(i, "props is a table of at most %d properties" % MAKE_MAX_PROPS)
    wire = {}
    for p, value in props.items():
        bad = cat.write_refusal(cls, p) if isinstance(p, str) else _bad(i, "a property name is text")
        if bad:
            return None, dict(bad, step=i)
        typ = cat.property_type(cls, p)
        if str(typ).startswith("Class."):
            if value is None:
                wire[p] = {"$t": "nil"}
                continue
            parts = V.split_path(value)
            if not parts:
                return None, _bad(i, "%s.%s holds an object: give its path, or null" % (cls, p), rule="type")
            if parts[0].startswith("@") and parts[0] not in ROOTS:
                return None, _bad(i, "%s.%s: %s is not a play root" % (cls, p, parts[0]), rule="type")
            wire[p] = {"$t": "Instance", "path": parts, "play": True}
            continue
        x, why = V.encode(typ, value, cat.enums)
        if why:
            return None, _bad(i, "%s.%s: %s" % (cls, p, why), rule="type")
        wire[p] = x
    return wire, None


def _set(cat, step, i, side, resolve, w):
    """set {target, class, props}: properties written in play (a camera made Scriptable, the cursor locked)."""
    parts, cls, bad = _target(step, i, side, resolve)
    if bad:
        return None, bad
    if cls not in cat.classes:
        return None, dict(cat.unknown_class(cls), step=i)
    props = step.get("props")
    if not isinstance(props, dict) or not props:
        return None, _bad(i, "set needs props: the properties to write")
    wire, bad = _encode_props(cat, cls, props, i)
    if bad:
        return None, bad
    w.update(target=parts, props=wire, **{"class": cls})
    if step.get("runtime") is True:
        w["runtime"] = True
    return w, None


def _make(cat, step, i, resolve, w):
    """make {class, parent, name, props, near}: a creatable class, props each writable and typed by the catalog."""
    cls, name = step.get("class"), step.get("name")
    if not isinstance(cls, str) or cat.create_refusal(cls):
        return None, dict(cat.create_refusal(cls) if isinstance(cls, str) else _bad(i, "make needs class"), step=i,
                          ran=False, ok=False)
    if not isinstance(name, str) or not name or len(name) > 64:
        return None, _bad(i, "make needs name: 1-64 characters (the steps after it find the object by it)")
    parent = V.split_path(step.get("parent"))
    if not parent:
        return None, _bad(i, "make needs parent: a play root path, a path in the place, or one made in play (runtime)")
    if not parent[0].startswith("@") and step.get("runtime") is not True:
        _pc, err = resolve(parent)
        if err:
            return None, _bad(i, "parent %s does not resolve in Edit: %s (give runtime true for a parent made in play)" % (
                V.join_path(parent), err.get("error")), rule=err.get("refusedRule") or "not-found")
    elif parent[0].startswith("@") and parent[0] not in ROOTS:
        return None, _bad(i, "%s is not a play root; the roots are %s" % (parent[0], ", ".join(ROOTS)))
    wire, bad = _encode_props(cat, cls, step.get("props") or {}, i)
    if bad:
        return None, bad
    w.update(**{"class": cls, "name": name, "parent": parent, "props": wire})
    if step.get("runtime") is True:
        w["runtime"] = True
    if step.get("near") is not None:
        if not cat.is_a(cls, "BasePart"):
            return None, _bad(i, "near places a part; %s is not a BasePart" % cls)
        n, why = V.encode("Vector3", step["near"], cat.enums)
        if why:
            return None, _bad(i, "near: %s" % why, rule="type")
        w["near"] = n
    return w, None


def _call_cause(cat, step, i, side, resolve, w, read_only):
    """call {target, class, method, args, check}: one PLAY_CALLS row, its arguments typed by the catalog."""
    parts, cls, bad = _target(step, i, side, resolve)
    if bad:
        return None, bad
    if cls not in cat.classes:
        return None, dict(cat.unknown_class(cls), step=i)
    m = step.get("method")
    if not isinstance(m, str) or not m:
        return None, _bad(i, "call needs method: a member of PLAY_CALLS")
    bad = cat.read_refusal(cls, m, kind="F")
    if bad:
        return None, dict(bad, step=i)
    owner, row = cat.find_member(cls, m)
    key = "%s.%s" % (owner, m)
    spec = PLAY_CALLS.get(key)
    if spec is None:
        return None, _bad(i, "%s is not in the play call list (functions are opt-in rows, never called blind): %s" % (
            key, ", ".join(sorted(PLAY_CALLS))), rule="play-write")
    if side not in spec["on"]:
        return None, _bad(i, "%s runs on the %s" % (key, " or ".join(spec["on"])))
    params = row[5]
    args = step.get("args", [])
    tup = bool(params) and params[-1][1] == "Tuple"
    fixed = params[:-1] if tup else params
    if not isinstance(args, list) or (len(args) > len(fixed) and not tup) or len(args) > len(fixed) + 8:
        return None, _bad(i, "%s takes %d arguments%s" % (key, len(fixed), " and up to 8 more" if tup else ""), rule="type")
    for p in fixed[len(args):]:
        if p[2] is None and not str(p[1]).endswith("?"):
            return None, _bad(i, "%s needs %s (%s)" % (key, p[0], p[1]), rule="type")
    wire, bad = _encode_args(cat, fixed, args[:len(fixed)], i, side)
    if bad:
        return None, bad
    for x in args[len(fixed):]:
        # a Tuple's values: plain values, or typed ones ({"$t": "EnumItem", "enum": "KeyCode", "v": "F"})
        e, why = V.encode("Variant", x, cat.enums)
        if why:
            return None, _bad(i, "%s: %s" % (params[-1][0], why), rule="type")
        wire.append(e)
    w.update(target=parts, method=m, args=wire, argc=len(wire), member=key, **{"class": cls})
    if step.get("runtime") is True:
        w["runtime"] = True
    chk = step.get("check")
    if chk is not None:
        if not spec.get("check"):
            return None, _bad(i, "%s is proven by %s, not by a check" % (key, ", ".join(spec["events"]) or "nothing"))
        cparts, ccls = parts, cls
        if isinstance(chk, dict) and chk.get("target") is not None:
            cparts, ccls, bad = _target(dict(chk, runtime=chk.get("runtime", step.get("runtime"))), i, side, resolve)
            if bad:
                return None, bad
        if not isinstance(chk, dict) or len([k for k in ("prop", "call") if k in chk]) != 1:
            return None, _bad(i, "check is {prop} or {call, args}, with target optional")
        c = {"target": cparts, "class": ccls}
        if "prop" in chk:
            bad = cat.read_refusal(ccls, chk["prop"], kind="P") if isinstance(chk["prop"], str) else _bad(i, "prop is a name")
            if bad:
                return None, dict(bad, step=i)
            c["prop"] = chk["prop"]
        else:
            g = chk["call"]
            bad = cat.read_refusal(ccls, g, kind="F") if isinstance(g, str) else _bad(i, "check call is a name")
            if bad:
                return None, dict(bad, step=i)
            _go, grow = cat.find_member(ccls, g)
            if not read_only(g, _go):
                return None, _bad(i, "check call %s is not a read by name" % g, rule="play-write")
            gw, bad = _encode_args(cat, grow[5], chk.get("args", []), i, side)
            if bad:
                return None, bad
            c.update(call=g, args=gw, argc=len(gw))
        mc = chk.get("min_change")
        if mc is not None:
            # a number or vector that drifts anyway (a part settling) must move at least this far to count
            if not isinstance(mc, (int, float)) or isinstance(mc, bool) or not mc > 0:
                return None, _bad(i, "min_change is a number above 0")
            c["minChange"] = mc
        w["check"] = c
    elif spec.get("check") and not spec.get("events"):
        return None, _bad(i, "%s is proven by a check: give check {prop} or {call, args}" % key)
    return w, None


def units(step):
    return len(step.get("props") or []) if step["op"] == "read" else (0 if step["op"] == "wait" else 1)


def split(steps, limit=MAX_UNITS_PER_RUN):
    """Runs of at most limit units each. Watches, causes and waits stay together in the first run, in plan order, with
    the reads and calls between them: a cause is never split from the watch it should raise. Only reads and calls after
    the last watch, cause or wait move to later runs."""
    last_bound = max([n for n, s in enumerate(steps) if s["op"] in ("watch", "cause", "wait")] or [-1])
    head, tail = steps[:last_bound + 1], steps[last_bound + 1:]
    runs, cur, n = [], list(head), sum(units(s) for s in head)
    for s in tail:
        u = units(s)
        if cur and n + u > limit:
            runs.append(cur)
            cur, n = [], 0
        cur.append(s)
        n += u
    runs.append(cur)
    return runs


ALREADY_MS = 2500      # how far before a cause firedBefore looks (RB2247: health regenerating fires about once a second)
CAUSE_SECONDS = 3     # what one cause may take: a key held, a prompt's hold, a touch settling (a hold longer is waited out)


def window_needed(steps):
    """Seconds the play window must last for each side's waits and causes to run, plus 2 for the last firings."""
    need = 0
    for side in SIDES:
        mine = [s for s in steps if s["on"] == side]
        need = max(need, sum(s.get("seconds", 0) for s in mine if s["op"] == "wait")
                   + sum(CAUSE_SECONDS + s.get("seconds", 0) for s in mine if s["op"] == "cause"))
    return need + 2


def needs_clients(steps):
    return any(s["on"] == "client" for s in steps)


def needs_remote(steps):
    return any((s["op"] == "cause" and s["cause"] == "remote" and not s.get("unreliable"))
               or s.get("target") == ["ReplicatedStorage", REMOTE_NAME] for s in steps)


def needs_unreliable(steps):
    return any((s["op"] == "cause" and s["cause"] == "remote" and s.get("unreliable"))
               or s.get("target") == ["ReplicatedStorage", UNRELIABLE_NAME] for s in steps)


def needs_plugin(steps):
    """The plugin version the plan's causes need (0.17.13 for the first ten)."""
    return max([CAUSES_PLUGIN.get(s.get("cause"), (0, 17, 13)) for s in steps if s["op"] == "cause"]
               + [(0, 17, 19) for s in steps if s.get("runtime") or s.get("early")
                  or (s.get("target") or [None])[0] in ("@input", "@track")
                  or (s.get("cause") == "move_to" and s.get("near") is not None)
                  or (s.get("cause") == "equip" and s.get("target") is not None)
                  or (s.get("op") == "cause" and s.get("cause") == "remote" and (s.get("on") == "server"
                                                                                or s.get("unreliable")))]
               + [(0, 17, 13)])


def _side_rows(side_result, side, player=None):
    if not isinstance(side_result, dict):
        return [], [], []
    tag = lambda r: dict(r, on=side, **({"player": player} if player else {}))  # noqa: E731
    return ([tag(r) for r in side_result.get("steps") or [] if isinstance(r, dict)],
            [tag(r) for r in side_result.get("watches") or [] if isinstance(r, dict)],
            [tag(r) for r in side_result.get("causes") or [] if isinstance(r, dict)])


def collect(r):
    """The members answers of one run: the server's (r.members) and each client's (r.clients[k].members)."""
    sides = []
    if isinstance(r.get("members"), dict):
        sides.append(("server", r["members"], None))
    for c in r.get("clients") or []:
        if isinstance(c, dict) and isinstance(c.get("members"), dict):
            sides.append(("client", c["members"], c.get("player") or c.get("localPlayer")))
    return sides


def _distance(a, b):
    """How far two wire numbers or vectors are apart; 0 when they are not both numbers or same-length vectors."""
    a, b = V.decode(a), V.decode(b)
    if isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b):
        a, b = [a[k] for k in sorted(a)], [b[k] for k in sorted(b)]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b)
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and all(
            isinstance(x, (int, float)) for x in a + b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
    return 0


def cause_proofs(c, step):
    """The members a cause's own calls proved: each whose proving event fired in the cause's window and not in the
    ALREADY_MS before it, or (a call with check) whose check read back changed. Nothing for a cause that failed."""
    if not c.get("ok") or not step:
        return []
    kind = step.get("cause")
    if kind == "call":
        spec = PLAY_CALLS.get(step.get("member")) or {}
        rows = {step.get("member"): tuple(spec.get("events") or ())}
    elif kind == "remote":
        rows = REMOTE_PROVES.get((step.get("on"), bool(step.get("unreliable"))), {})
    else:
        rows = CAUSE_PROVES.get(kind, {})
    fresh = {r.get("member") for r in c.get("raised") or [] if r.get("member") and not r.get("firedBefore")}
    out = [m for m, evs in rows.items() if fresh & set(evs)]
    chk = (c.get("detail") or {}).get("check") if isinstance(c.get("detail"), dict) else None
    changed = isinstance(chk, dict) and chk.get("ok") is True and "before" in chk and "after" in chk \
        and chk["before"] != chk["after"]
    mc = (step.get("check") or {}).get("minChange") if kind == "call" else None
    if changed and mc is not None:
        changed = _distance(chk["before"], chk["after"]) >= mc
    if changed and kind == "call" and step.get("check") and step.get("member") not in out:
        out.append(step.get("member"))
    if changed:
        # text_input: the focused TextBox's Text before and after; animate: the track's IsPlaying after Play
        out += [m for m in CHECK_PROVES.get(kind, ()) if m not in out]
    return out


def judge(cat, plan, results, trusted):
    """One answer from every run's results: steps in plan order, watches with their firings, causes each with what it
    raised (the watched events, either side, that fired between it and the next cause or the end; raisedNothing when
    none), every planned step that no side answered named in missing, and proved - the members the record may take -
    only when trusted."""
    steps, watches, causes, ends = [], [], [], []
    for r in results:
        for side, res, player in collect(r):
            s, w, c = _side_rows(res, side, player)
            steps += s
            watches += w
            causes += c
            if isinstance(res.get("endedAt"), (int, float)):
                ends.append(res["endedAt"])
    answered = {r.get("i") for r in steps} | {r.get("i") for r in watches} | {r.get("i") for r in causes}
    missing = [s["i"] for s in plan if s["op"] != "wait" and s["i"] not in answered]
    end = max(ends) if ends else None
    causes.sort(key=lambda c: (c.get("at") or 0, c.get("i") or 0))

    def fired(w, lo, hi):
        return sum(1 for f in w.get("firings") or [] if isinstance(f.get("at"), (int, float))
                   and (lo is None or f["at"] >= lo) and (hi is None or f["at"] < hi))
    by_i = {s.get("i"): s for s in plan}
    for k, c in enumerate(causes):
        # A cause's window is [its start, the next cause's start). Time alone cannot tell a cause's event from one going
        # on anyway (a Humanoid's health regenerating about once a second, live RB2247), so each event that fired in the
        # window says how often it fired in the ALREADY_MS before the cause too (firedBefore). With expect - the events
        # the plan says this cause should raise - raisedNothing is whether none of those fired; without, whether nothing
        # at all fired in the window.
        lo = c.get("at")
        hi = causes[k + 1].get("at") if k + 1 < len(causes) else (end + 1 if end is not None else None)
        raised = []
        for w in watches:
            n = fired(w, lo, hi) if isinstance(lo, (int, float)) else 0
            if n:
                wc = w.get("className") or w.get("class")
                owner = cat.find_member(wc, w.get("event"))[0] if cat is not None and wc in cat.classes else None
                raised.append({"step": w.get("i"), "on": w.get("on"), "event": "%s.%s" % (w.get("class"), w.get("event")),
                               "member": "%s.%s" % (owner, w.get("event")) if owner else None,
                               "count": n, "firedBefore": fired(w, lo - ALREADY_MS, lo)})
        c["raised"] = raised
        expect = (by_i.get(c.get("i")) or {}).get("expect")
        if expect:
            got = {r["event"] for r in raised}
            c["expected"] = [{"event": e, "fired": e in got} for e in expect]
            c["raisedNothing"] = not any(x["fired"] for x in c["expected"])
        else:
            c["raisedNothing"] = not raised
    proved = []
    if trusted:
        seen = set()

        def add(cls, name, how):
            owner, row = cat.find_member(cls, name) if cls in cat.classes else (None, None)
            key = "%s.%s" % (owner, name)
            if row is not None and owner and C.is_counted(row) and (key, how) not in seen:
                seen.add((key, how))
                proved.append({"member": key, "how": how})
        for s in steps:
            cls = s.get("className") or s.get("class")
            if not s.get("ok"):
                continue
            if s.get("op") == "read":
                # Studio writes an empty table as [] (tools._obj): no values is no proof
                for p, v in (s.get("values") if isinstance(s.get("values"), dict) else {}).items():
                    # a value the plugin could only give as text ({"$t", "str"}) did not decode: not a proof
                    if not (isinstance(v, dict) and "str" in v and "v" not in v):
                        add(cls, p, "play-read")
            elif s.get("op") == "call":
                add(cls, s.get("method"), "play-call")
        for w in watches:
            if w.get("ok") and w.get("firings"):
                add(w.get("className") or w.get("class"), w.get("event"), "play-event")
        ran = []                     # (side, member) of call causes that ran ok, in time order
        for c in causes:
            step = by_i.get(c.get("i")) or {}
            keys = cause_proofs(c, step)
            for key in keys:
                add(*key.split(".", 1), "play-call-verified")
                # a call whose proof needs another to have run first (data from a profiler that was started)
                for need in (PLAY_CALLS.get(key) or {}).get("requires") or ():
                    if (c.get("on"), need) in ran:
                        add(*need.split(".", 1), "play-call-verified")
            if c.get("ok") and step.get("cause") == "call":
                ran.append((c.get("on"), step.get("member")))
    decoded = []
    for s in steps:
        d = dict(s)
        if isinstance(d.get("values"), dict):
            d["values"] = {k: V.decode(v) for k, v in d["values"].items()}
        if "returns" in d:
            d["returns"] = [V.decode(x) for x in d.get("returns") or []]
        decoded.append(d)
    for w in watches:
        w["firings"] = [dict(f, args=[V.decode(x) for x in f.get("args") or []]) for f in w.get("firings") or []]
    decoded.sort(key=lambda s: (s.get("i") or 0, s.get("on") or ""))
    return {"steps": decoded, "watches": watches, "causes": causes, "missing": missing, "proved": proved,
            "trusted": bool(trusted)}
