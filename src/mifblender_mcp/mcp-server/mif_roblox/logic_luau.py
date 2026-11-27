"""Luau sources for the game-logic checks (domain_logic.py): the DataStore stand-in, the census and self-test runners,
the remote fuzzer, and the save, action, once-per-player prompt and economy templates rbx_script_scaffold writes.

Kept here as fixed text so each one is reviewed once and checked offline (a dev test compiles them when
a Luau analyzer is installed). Templates take their name through the token __NAME__ only; nothing a caller sends is ever
pasted into source: settings travel as attributes (JSON strings) that the scripts read.

The stand-in seam: a template asks for its data stores through `dataStores()`, which in Studio (RunService:IsStudio())
answers Workspace.MifScratch.DataStoreStandIn when that ModuleScript is there, and DataStoreService everywhere else. A
live server never sees the stand-in. A game's own save code gets the same seam by adding that function.
"""
import re

# ---------------------------------------------------------------- the one print channel
# Every probe script and self-test hands its JSON to the MCP side through its Output the same way (found 2026-11-22: the
# runner and the fuzzer cut raw JSON every 800 bytes, which split a UTF-8 character across two prints, RB2070, and the
# Output drops a backslash, so an escaped quote in a name broke the JSON, RB1451). Every byte outside letters, digits,
# space and , : { } [ ] . _ - " goes as ~ and two hex digits - domain_controller.CLIENT_EMIT's escape - and that ASCII
# text is cut every 800 characters, each part after its "MARK tag i/n " prefixed ~ so domain_logic.marked knows the form
# (an older script's raw part never starts with ~: JSON does not). The sources below hold markers, expanded once at
# import: "--@emitfn" becomes emit(tag, value) (MARK and HttpService in scope), "--@emit LABEL" a block printing j.
EMIT_ESCAPE = '[^%w ,:{}%[%]%._%-"]'
EMIT_FN = r'''local function emit(tag, value)
	local ok, j = pcall(function() return HttpService:JSONEncode(value) end)
	if not ok then j = HttpService:JSONEncode({ encodeError = tostring(j) }) end
	local text = string.gsub(j, '__ESC__', function(c) return string.format("~%02x", string.byte(c)) end)
	local n = math.max(1, math.ceil(#text / 800))
	for i = 1, n do
		print(string.format("%s %s %d/%d ~%s", MARK, tag, i, n, string.sub(text, (i - 1) * 800 + 1, i * 800)))
	end
end
'''.replace("__ESC__", EMIT_ESCAPE)
EMIT_BLOCK = r'''do -- MifBridge's one print channel (logic_luau.EMIT_FN): ~XX-escaped JSON in 800-character parts
	local text = string.gsub(j, '__ESC__', function(c) return string.format("~%02x", string.byte(c)) end)
	local n = math.max(1, math.ceil(#text / 800))
	for i = 1, n do
		print(string.format("%s %d/%d ~%s", "__LABEL__", i, n, string.sub(text, (i - 1) * 800 + 1, i * 800)))
	end
end
'''.replace("__ESC__", EMIT_ESCAPE)


def with_emitter(src):
    """src with its emit markers expanded: --@emitfn to EMIT_FN, --@emit LABEL to EMIT_BLOCK printing j under LABEL."""
    src = src.replace("--@emitfn\n", EMIT_FN)

    def block(m):
        ind = m.group(1)
        return "".join(ind + ln + "\n" for ln in EMIT_BLOCK.replace("__LABEL__", m.group(2)).splitlines())
    return re.sub(r"(?m)^([ \t]*)--@emit (\S+ \S+)\n", block, src)


# ---------------------------------------------------------------- the DataStore stand-in (ModuleScript)
# Attributes it reads: Schedule (JSON list of events), Seed (JSON {store: {key: value}}).
# An event: {op: "UpdateAsync"|"SetAsync"|"GetAsync"|"RemoveAsync"|"IncrementAsync"|"*", store: name|"*", key: key|"*",
#            nth: first matching call it acts on (1-based, default 1), count: how many matching calls from nth (default 1,
#            0 = every one after), after / before: seconds since the stand-in started (a window, either or both),
#            do: "fail"|"stall"|"lock"|"unlock", seconds: a stall's length, then: "fail" to fail after a stall,
#            field: the lock's field (default "Lock"), session: the foreign session id (default "MifForeignServer")}
STAND_IN = r'''-- MifBridge DataStore stand-in: written by rbx_play_probe (fault_store) for one play test, removed after.
-- Answers like DataStoreService:GetDataStore(name) in memory, and fails, stalls or holds a foreign lock on cue.
local HttpService = game:GetService("HttpService")
local M = { isStandIn = true }
local t0 = os.clock()
local okS, schedule = pcall(function() return HttpService:JSONDecode(script:GetAttribute("Schedule") or "[]") end)
if not okS or type(schedule) ~= "table" then schedule = {} end
local okD, seed = pcall(function() return HttpService:JSONDecode(script:GetAttribute("Seed") or "{}") end)
if not okD or type(seed) ~= "table" then seed = {} end
local data = {}          -- store name -> key -> value (a JSON copy, as a real store serialises)
local seen = {}          -- event index -> matching calls so far
local fired = {}         -- event index -> times it acted
local calls = {}         -- op -> count
local inFlight = {}      -- store/key -> writes running now
local maxInFlight = {}   -- store/key -> most at once
local log = {}

local function copy(v)
	if v == nil then return nil end
	return HttpService:JSONDecode(HttpService:JSONEncode({ v = v })).v
end
for storeName, keys in pairs(seed) do
	data[storeName] = {}
	if type(keys) == "table" then
		for k, v in pairs(keys) do data[storeName][tostring(k)] = copy(v) end
	end
end
local function note(entry)
	if #log < 300 then table.insert(log, entry) end
end
local function matching(op, storeName, key)
	local now = os.clock() - t0
	local hits = {}
	for i, ev in ipairs(schedule) do
		if (ev.op == nil or ev.op == "*" or ev.op == op) and (ev.store == nil or ev.store == "*" or ev.store == storeName)
			and (ev.key == nil or ev.key == "*" or ev.key == key) then
			if (ev.after == nil or now >= ev.after) and (ev.before == nil or now < ev.before) then
				seen[i] = (seen[i] or 0) + 1
				local nth, count = ev.nth or 1, ev.count or 1
				if seen[i] >= nth and (count == 0 or seen[i] < nth + count) then
					fired[i] = (fired[i] or 0) + 1
					table.insert(hits, ev)
				end
			end
		end
	end
	return hits
end
local function lockOf(ev)
	return ev.field or "Lock", { Session = ev.session or "MifForeignServer", Time = os.time() }
end
-- Runs the schedule for one call: locks and unlocks act on the stored value; a stall waits; a fail raises.
local keyCalls = {}      -- store/key/op -> count
local function cue(op, storeName, key)
	calls[op] = (calls[op] or 0) + 1
	keyCalls[storeName .. "/" .. key .. "/" .. op] = (keyCalls[storeName .. "/" .. key .. "/" .. op] or 0) + 1
	local store = data[storeName]
	for _, ev in ipairs(matching(op, storeName, key)) do
		note({ t = math.floor((os.clock() - t0) * 1000) / 1000, op = op, store = storeName, key = key, did = ev["do"] })
		if ev["do"] == "lock" or ev["do"] == "unlock" then
			local field, lock = lockOf(ev)
			local v = store[key]
			if type(v) ~= "table" then v = {} end
			v[field] = if ev["do"] == "lock" then lock else nil
			store[key] = v
		elseif ev["do"] == "stall" then
			task.wait(ev.seconds or 5)
			if ev["then"] == "fail" then
				error("MifBridge stand-in: scheduled failure after a stall (" .. op .. " " .. tostring(key) .. ")", 0)
			end
		elseif ev["do"] == "fail" then
			error(ev.message or ("MifBridge stand-in: scheduled failure (" .. op .. " " .. tostring(key) .. ")"), 0)
		end
	end
end
local function enter(storeName, key)
	local id = storeName .. "/" .. key
	inFlight[id] = (inFlight[id] or 0) + 1
	maxInFlight[id] = math.max(maxInFlight[id] or 0, inFlight[id])
	return id
end
local function leave(id)
	inFlight[id] -= 1
end
-- Live, UpdateAsync refuses a transform that yields ("Callbacks cannot yield"): a DataStore, HTTP or task.wait call
-- inside it. Calling the transform directly ran such a callback without complaint, so code reading another store inside
-- it passed every Studio test and failed every live call. It runs in a coroutine here; one that is not dead after its
-- first resume yielded, is closed so it never resumes, and the call fails as live does, writing nothing.
local yielded = 0
local function runTransform(transform, old, storeName, key)
	local co = coroutine.create(transform)
	local okR, new = coroutine.resume(co, old, nil)
	if not okR then
		error(new, 0)
	end
	if coroutine.status(co) ~= "dead" then
		pcall(coroutine.close, co)
		yielded += 1
		note({ t = math.floor((os.clock() - t0) * 1000) / 1000, op = "UpdateAsync", store = storeName, key = key, did = "yieldRefused" })
		error("Callbacks cannot yield", 0)
	end
	return new
end

function M:GetDataStore(name, scope)
	local storeName = tostring(name) .. (if scope ~= nil and scope ~= "global" then "/" .. tostring(scope) else "")
	data[storeName] = data[storeName] or {}
	local store = data[storeName]
	local S = {}
	function S:GetAsync(key)
		key = tostring(key)
		cue("GetAsync", storeName, key)
		return copy(store[key]), nil
	end
	function S:SetAsync(key, value)
		key = tostring(key)
		local id = enter(storeName, key)
		local ok, err = pcall(cue, "SetAsync", storeName, key)
		if ok then store[key] = copy(value) end
		leave(id)
		if not ok then error(err, 0) end
		return "MifStandInVersion"
	end
	function S:UpdateAsync(key, transform)
		key = tostring(key)
		local id = enter(storeName, key)
		local ok, err = pcall(cue, "UpdateAsync", storeName, key)
		local result = nil
		if ok then
			ok, err = pcall(function()
				local new = runTransform(transform, copy(store[key]), storeName, key)
				if new ~= nil then
					store[key] = copy(new)
				end
				result = copy(if new ~= nil then new else nil)
			end)
		end
		leave(id)
		if not ok then error(err, 0) end
		return result, nil
	end
	function S:RemoveAsync(key)
		key = tostring(key)
		local id = enter(storeName, key)
		local ok, err = pcall(cue, "RemoveAsync", storeName, key)
		local old = store[key]
		if ok then store[key] = nil end
		leave(id)
		if not ok then error(err, 0) end
		return copy(old), nil
	end
	function S:IncrementAsync(key, delta)
		key = tostring(key)
		local id = enter(storeName, key)
		local ok, err = pcall(cue, "IncrementAsync", storeName, key)
		if ok then store[key] = (tonumber(store[key]) or 0) + (delta or 1) end
		leave(id)
		if not ok then error(err, 0) end
		return store[key]
	end
	return S
end
function M:GetRequestBudgetForRequestType()
	return 100
end
-- For self-tests and the census: read or change a stored value directly, add an event, and the report.
function M.peek(storeName, key)
	return copy((data[storeName] or {})[tostring(key)])
end
function M.setLock(storeName, key, session, field)
	data[storeName] = data[storeName] or {}
	local v = data[storeName][tostring(key)]
	if type(v) ~= "table" then v = {} end
	v[field or "Lock"] = if session then { Session = session, Time = os.time() } else nil
	data[storeName][tostring(key)] = v
end
function M.schedule(ev)
	table.insert(schedule, ev)
	return #schedule
end
function M.callsFor(storeName, key, op)
	return keyCalls[storeName .. "/" .. tostring(key) .. "/" .. op] or 0
end
function M.maxConcurrent(storeName, key)
	return maxInFlight[storeName .. "/" .. tostring(key)] or 0
end
function M.report()
	local events = {}
	for i, ev in ipairs(schedule) do
		table.insert(events, { index = i, op = ev.op, key = ev.key, ["do"] = ev["do"], matched = seen[i] or 0, acted = fired[i] or 0 })
	end
	local most = 0
	for _, n in pairs(maxInFlight) do most = math.max(most, n) end
	return { calls = calls, events = events, log = log, maxConcurrentWrites = most, yieldedTransforms = yielded }
end
return M
'''

# ---------------------------------------------------------------- the census and self-test runner (server Script)
# Attributes on the runner: Census (JSON {module: [path], fn, per: "player"|"all"} or ""), SelfTests (JSON [names]),
# Players (how many to wait for), Before / After (seconds from start), Mark (the print tag).
RUNNER = r'''-- MifBridge census and self-test runner: written by rbx_play_probe for one play test, removed after. Read only:
-- it calls the census function the caller named and prints what it returns, with the stand-in's report.
local RunService = game:GetService("RunService")
if not RunService:IsStudio() then return end  -- left in a saved place, it does nothing live
local HttpService = game:GetService("HttpService")
local Players = game:GetService("Players")
local MARK = script:GetAttribute("Mark") or "MIFLOGIC"
local t0 = os.clock()
--@emitfn
local function find(path)
	local cur = game
	for i, name in ipairs(path) do
		cur = (i == 1 and game:FindService(name)) or (cur and cur:FindFirstChild(name))
		if cur == nil then return nil end
	end
	return cur
end
local function plain(v, depth)
	depth = depth or 0
	local t = typeof(v)
	if t == "number" then
		if v ~= v then return "NaN" end
		if v == math.huge then return "inf" end
		if v == -math.huge then return "-inf" end
		return v
	end
	if t == "string" or t == "boolean" then return v end
	if t == "Instance" then return v:GetFullName() end
	if t == "table" and depth < 6 then
		local out = {}
		local n = 0
		for k, x in pairs(v) do
			n += 1
			if n > 2000 then break end
			out[tostring(k)] = plain(x, depth + 1)
		end
		if #v > 0 and #v == n then
			local arr = {}
			for i = 1, #v do arr[i] = plain(v[i], depth + 1) end
			return arr
		end
		return out
	end
	return tostring(v)
end
local okC, census = pcall(function() return HttpService:JSONDecode(script:GetAttribute("Census") or "") end)
if not okC then census = nil end
local okT, tests = pcall(function() return HttpService:JSONDecode(script:GetAttribute("SelfTests") or "[]") end)
if not okT or type(tests) ~= "table" then tests = {} end
local want = script:GetAttribute("Players") or 0
local function take(label)
	if type(census) ~= "table" then return end
	local snap = { at = label, t = os.clock() - t0, owners = {} }
	local mod = find(census.module or {})
	if mod == nil or not mod:IsA("ModuleScript") then
		snap.error = "census module not found: " .. table.concat(census.module or {}, ".")
		emit("census", snap)
		return
	end
	local okR, M = pcall(require, mod)
	local fn = okR and type(M) == "table" and M[census.fn or "census"]
	if type(fn) ~= "function" then
		snap.error = if okR then "the census module has no function " .. tostring(census.fn) else "require failed: " .. tostring(M)
		emit("census", snap)
		return
	end
	if census.per == "all" then
		local okF, res = pcall(fn)
		if okF and type(res) == "table" then
			for owner, items in pairs(res) do snap.owners[tostring(owner)] = plain(items) end
		else
			snap.error = "census(): " .. tostring(res)
		end
	else
		for _, p in ipairs(Players:GetPlayers()) do
			local okF, res = pcall(fn, p)
			snap.owners[p.Name] = if okF then plain(res) else { error = tostring(res) }
		end
	end
	emit("census", snap)
end
task.wait(1) -- the probe's own Output capture starts with its script; print after it
local deadline = os.clock() + 30
while #Players:GetPlayers() < want and os.clock() < deadline do task.wait(0.2) end
local before = script:GetAttribute("Before") or 2
while os.clock() - t0 < before do task.wait(0.1) end
take("before")
local scratch = workspace:FindFirstChild("MifScratch")
for _, name in ipairs(tests) do
	local marker = Instance.new("Folder")
	marker.Name = "SelfTest_" .. tostring(name)
	marker.Parent = scratch
end
local after = script:GetAttribute("After") or 10
while os.clock() - t0 < after do task.wait(0.1) end
take("after")
local standIn = scratch and scratch:FindFirstChild("DataStoreStandIn")
if standIn then
	local okS, S = pcall(require, standIn)
	emit("standIn", if okS then S.report() else { error = tostring(S) })
end
emit("done", { t = os.clock() - t0 })
'''

# ---------------------------------------------------------------- remote fuzz: the server half (Script)
# Attributes: Config (JSON {roots: [[path]...], census: {module, fn}?, watch: [[path]...], ignore: [keys], wait, controls}),
# Mark. Children it makes: MifFuzzMark (RemoteFunction). It names the fuzzing player in the folder's Fuzzer attribute.
FUZZ_SERVER = r'''-- MifBridge remote fuzz, server half: written by rbx_remote_fuzz for one play test, removed after. It only reads
-- state: before and after each forged call it snapshots leaderstats, player attributes, the watched objects and the
-- census the caller named, and prints what changed.
local RunService = game:GetService("RunService")
if not RunService:IsStudio() then return end  -- left in a saved place, it does nothing live
local HttpService = game:GetService("HttpService")
local Players = game:GetService("Players")
local folder = script.Parent
local MARK = folder:GetAttribute("Mark") or "MIFFUZZ"
local cfg = HttpService:JSONDecode(folder:GetAttribute("Config") or "{}")
local t0 = os.clock()
--@emitfn
local function find(path)
	local cur = game
	for i, name in ipairs(path) do
		cur = (i == 1 and game:FindService(name)) or (cur and cur:FindFirstChild(name))
		if cur == nil then return nil end
	end
	return cur
end
local function val(v)
	local t = typeof(v)
	if t == "number" then
		if v ~= v then return "NaN" end
		if v == math.huge then return "inf" end
		if v == -math.huge then return "-inf" end
		return v
	end
	if t == "string" or t == "boolean" or t == "nil" then return v end
	if t == "Instance" then return v:GetFullName() end
	return tostring(v)
end
local censusFn = nil
if type(cfg.census) == "table" then
	local mod = find(cfg.census.module or {})
	if mod and mod:IsA("ModuleScript") then
		local ok, M = pcall(require, mod)
		if ok and type(M) == "table" and type(M[cfg.census.fn or "census"]) == "function" then
			censusFn = M[cfg.census.fn or "census"]
		end
	end
end
local function snapshot()
	local s = {}
	for _, p in ipairs(Players:GetPlayers()) do
		local ls = p:FindFirstChild("leaderstats")
		if ls then
			for _, v in ipairs(ls:GetChildren()) do
				if v:IsA("ValueBase") then s[p.Name .. ".leaderstats." .. v.Name] = val(v.Value) end
			end
		end
		for k, v in pairs(p:GetAttributes()) do s[p.Name .. "@" .. k] = val(v) end
		if censusFn then
			local ok, res = pcall(censusFn, p)
			if ok and type(res) == "table" then
				local n = 0
				for k, v in pairs(res) do
					n += 1
					if n > 500 then break end
					if type(k) == "number" then
						s[p.Name .. "#" .. tostring(v)] = (s[p.Name .. "#" .. tostring(v)] or 0) + 1
					else
						s[p.Name .. "#" .. tostring(k)] = val(v)
					end
				end
			else
				s[p.Name .. "#censusError"] = tostring(res)
			end
		end
	end
	for _, path in ipairs(cfg.watch or {}) do
		local o = find(path)
		local name = table.concat(path, ".")
		if o == nil then
			s[name] = "(missing)"
		else
			if o:IsA("ValueBase") then s[name] = val(o.Value) end
			for k, v in pairs(o:GetAttributes()) do s[name .. "@" .. k] = val(v) end
			s[name .. "#children"] = #o:GetChildren()
		end
	end
	return s
end
local ignore = {}
for _, k in ipairs(cfg.ignore or {}) do ignore[k] = true end
local function diff(a, b)
	local out, n = {}, 0
	for k, v in pairs(b) do
		if a[k] ~= v and not ignore[k] then
			out[k] = { a[k], v }
			n += 1
		end
	end
	for k, v in pairs(a) do
		if b[k] == nil and not ignore[k] then
			out[k] = { v, nil }
			n += 1
		end
	end
	return out, n
end
local mark = Instance.new("RemoteFunction")
mark.Name = "MifFuzzMark"
mark.Parent = folder
local last = nil
local drift = {}
local findings, cases, remotes = {}, 0, nil
local doneAt = nil
local clientSaid = {}
-- race (rbx_remote_fuzz race): one remote fired twice in one frame with valid arguments, the balance read around it
local race = nil
local function balance(player)
	local key = type(cfg.race) == "table" and cfg.race.balance or nil
	if type(key) ~= "string" then return nil end
	if string.sub(key, 1, 1) == "@" then
		local v = player:GetAttribute(string.sub(key, 2))
		return if type(v) == "number" then v else nil
	end
	local o = player
	for name in string.gmatch(key, "[^%.]+") do
		o = o and o:FindFirstChild(name)
	end
	return if o and o:IsA("ValueBase") and type(o.Value) == "number" then o.Value else nil
end
local function finish(info, why)
	if doneAt then return end
	doneAt = os.clock() - t0
	local driftList = {}
	for k in pairs(drift) do table.insert(driftList, k) end
	emit("result", { remotes = remotes or {}, cases = cases, findings = findings, drift = driftList,
		skipped = type(info) == "table" and info.skipped or nil, planned = type(info) == "table" and info.planned or nil,
		batches = type(info) == "table" and info.batches or nil,
		players = #Players:GetPlayers(), watched = cfg.watch or {}, census = censusFn ~= nil,
		censusAsked = type(cfg.census) == "table", t = doneAt, incomplete = why, client = clientSaid, race = race })
end
mark.OnServerInvoke = function(player, what, info)
	if what == "hello" or what == "crash" then
		table.insert(clientSaid, { player = player.Name, what = what, info = tostring(info) })
		emit(what, { player = player.Name, info = tostring(info), t = os.clock() - t0 })
		if what == "crash" and player.Name == folder:GetAttribute("Fuzzer") then
			finish(nil, "the fuzzing client stopped: " .. tostring(info))
		end
		return true
	end
	if player.Name ~= folder:GetAttribute("Fuzzer") then return false end
	if what == "raceBefore" then
		race = { before = balance(player), price = type(cfg.race) == "table" and cfg.race.price or nil }
		return true
	elseif what == "raceAfter" then
		task.wait(math.max(cfg.wait or 0.15, 1))     -- a handler that saves before it pays has time to land
		race = race or {}
		race.after = balance(player)
		if type(info) == "table" then
			race.remote, race.class, race.error, race.answers = info.remote, info.class, info.error, info.answers
		end
		return true
	end
	task.wait(cfg.wait or 0.15)
	local now = snapshot()
	if what == "start" then
		last = now
		remotes = info
		return true
	end
	local changed, n = diff(last or now, now)
	last = now
	if what == "control" then
		for k in pairs(changed) do drift[k] = true end
	elseif what == "batch" then
		-- every argument of one remote fired: answer whether anything but drift changed; the client then fires
		-- them one by one only for a remote that did
		for k in pairs(changed) do
			if not drift[k] then return "changed" end
		end
		return "clean"
	elseif what == "case" and type(info) == "table" then
		cases += 1
		if n > 0 then
			local keep = {}
			for k, v in pairs(changed) do
				if not drift[k] then keep[k] = v end
			end
			if next(keep) ~= nil and #findings < 200 then
				table.insert(findings, { remote = info.remote, class = info.class, arg = info.arg, changed = keep,
					returned = info.returned, error = info.error })
			end
		end
	elseif what == "done" then
		finish(info, nil)
	end
	return true
end
local tw = os.clock()
while #Players:GetPlayers() < 1 and os.clock() - tw < 30 do task.wait(0.1) end
local first = Players:GetPlayers()[1]
if first then folder:SetAttribute("Fuzzer", first.Name) end
-- a result even when the client never says done (it did not run, or it was cut short), before the probe ends
task.delay(math.max(1, (cfg.deadline or 40) - (os.clock() - t0)), function()
	finish(nil, if first then "the fuzzing client did not finish in time" else "no player joined")
end)
'''

# ---------------------------------------------------------------- remote fuzz: the client half (Script, RunContext Client)
FUZZ_CLIENT = r'''-- MifBridge remote fuzz, client half: written by rbx_remote_fuzz for one play test, removed after. The first
-- player's client fires every remote under the roots with forged arguments, each twice, and asks the server half to
-- compare state after each one.
local RunService = game:GetService("RunService")
if not RunService:IsStudio() then return end  -- left in a saved place, it never fires a remote live
local HttpService = game:GetService("HttpService")
local Players = game:GetService("Players")
local folder = script.Parent
local me = Players.LocalPlayer
local t0 = os.clock()
while folder:GetAttribute("Fuzzer") == nil and os.clock() - t0 < 30 do task.wait(0.1) end
if folder:GetAttribute("Fuzzer") ~= me.Name then return end
local cfg = HttpService:JSONDecode(folder:GetAttribute("Config") or "{}")
local mark = folder:WaitForChild("MifFuzzMark", 30)
if mark == nil then return end
mark:InvokeServer("hello", me.Name)
local budget = cfg.budget or 40
local okRun, why = pcall(function()
local function find(path)
	local cur = game
	for i, name in ipairs(path) do
		cur = (i == 1 and game:FindService(name)) or (cur and cur:WaitForChild(name, 5))
		if cur == nil then return nil end
	end
	return cur
end
local remotes = {}
for _, path in ipairs(cfg.roots or {}) do
	local root = find(path)
	if root then
		local list = root:GetDescendants()
		table.insert(list, 1, root)
		for _, d in ipairs(list) do
			if (d:IsA("RemoteEvent") or d:IsA("UnreliableRemoteEvent") or d:IsA("RemoteFunction"))
				and not d:IsDescendantOf(folder) and d.Name ~= "MifProbeEvent" then
				table.insert(remotes, d)
			end
		end
	end
end
local names = {}
for _, r in ipairs(remotes) do table.insert(names, { path = r:GetFullName(), class = r.ClassName }) end
mark:InvokeServer("start", names)
local controls = cfg.controls or 3
for _ = 1, controls do mark:InvokeServer("control") end
local other = nil
for _, p in ipairs(Players:GetPlayers()) do
	if p ~= me then
		other = p
		break
	end
end
local char = me.Character or me.CharacterAdded:Wait()
local part = char:FindFirstChildWhichIsA("BasePart")
local function nested() return { a = { b = { c = {} } } } end
local args = {
	{ "nothing", function() return {} end },
	{ "nil", function() return { nil } end, true },
	{ "NaN", function() return { 0 / 0 } end },
	{ "+inf", function() return { math.huge } end },
	{ "-inf", function() return { -math.huge } end },
	{ "huge", function() return { 2 ^ 53 } end },
	{ "negative", function() return { -1 } end },
	{ "big negative", function() return { -1e9 } end },
	{ "fraction", function() return { 0.5 } end },
	{ "empty table", function() return { {} } end },
	{ "table", function() return { { 1, 2, 3 } } end },
	{ "nested table", function() return { nested() } end },
	{ "string", function() return { "MifFuzz" } end },
	{ "long string", function() return { string.rep("x", 1000) } end },
	{ "boolean", function() return { true } end },
	{ "Vector3", function() return { Vector3.new(1e9, 1e9, 1e9) } end },
	{ "instance", function() return { part or workspace } end },
	{ "self", function() return { me } end },
	{ "another player", function() return if other then { other } else nil end },
	{ "NaN after a string", function() return { "MifFuzz", 0 / 0 } end },
}
local planned, skipped, cases = #remotes, {}, 0
local function fire(r, a)
	if r:IsA("RemoteFunction") then
		local done, ret, err = false, nil, nil
		task.spawn(function()
			local ok, res = pcall(function() return r:InvokeServer(table.unpack(a, 1, a.n or #a)) end)
			if ok then ret = res else err = tostring(res) end
			done = true
		end)
		local tw = os.clock()
		while not done and os.clock() - tw < 2 do task.wait() end
		if not done then err = "no answer in 2 s" end
		return ret, err
	end
	r:FireServer(table.unpack(a, 1, a.n or #a))
	return nil, nil
end
-- One remote at a time: every argument fired twice (a function's invocations all at once), one comparison; only a
-- remote that changed state is fired again argument by argument, so each change is pinned to its argument.
local function fireAll(r, list)
	if r:IsA("RemoteFunction") then
		local left = #list * 2
		for _, a in ipairs(list) do
			for _ = 1, 2 do
				task.spawn(function()
					pcall(function() r:InvokeServer(table.unpack(a, 1, a.n or #a)) end)
					left -= 1
				end)
			end
		end
		local tw = os.clock()
		while left > 0 and os.clock() - tw < 3 do task.wait() end
	else
		for _, a in ipairs(list) do
			r:FireServer(table.unpack(a, 1, a.n or #a))
			r:FireServer(table.unpack(a, 1, a.n or #a))
		end
	end
end
-- The race runs FIRST, before any forged call (found live 2026-11-26: run after the fuzz, a shop that yields before it
-- charges was still paying out the fuzz's forged string calls inside the race's window, so the count was inflated,
-- and the fuzz had already spent the balance the race needed)
if type(cfg.race) == "table" then
	local rr = find(cfg.race.remote or {})
	local a = cfg.race.args or {}
	local info = { remote = rr and rr:GetFullName() or table.concat(cfg.race.remote or {}, "."),
		class = rr and rr.ClassName or nil, answers = {} }
	mark:InvokeServer("raceBefore")
	if rr == nil or not (rr:IsA("RemoteEvent") or rr:IsA("RemoteFunction")) then
		info.error = "no RemoteEvent or RemoteFunction at " .. info.remote
	elseif rr:IsA("RemoteFunction") then
		-- task.spawn runs each call up to its InvokeServer at once, so both requests leave in this frame
		local left = 2
		for i = 1, 2 do
			task.spawn(function()
				local ok, res = pcall(function() return rr:InvokeServer(table.unpack(a)) end)
				info.answers[i] = if ok then tostring(res) else "error: " .. tostring(res)
				left -= 1
			end)
		end
		local tw = os.clock()
		while left > 0 and os.clock() - tw < 5 do task.wait() end
		if left > 0 then info.error = "a call had no answer in 5 s" end
	else
		rr:FireServer(table.unpack(a))
		rr:FireServer(table.unpack(a))
	end
	mark:InvokeServer("raceAfter", info)
end
for _, r in ipairs(remotes) do
	local list = {}
	for _, spec in ipairs(args) do
		local a = spec[2]()
		if a == nil then
			table.insert(skipped, r:GetFullName() .. " " .. spec[1] .. " (needs players 2)")
		else
			if spec[3] then a.n = 1 end
			table.insert(list, a)
		end
	end
	if os.clock() - t0 > budget then
		table.insert(skipped, r:GetFullName() .. " (all arguments)")
		continue
	end
	fireAll(r, list)
	cases += 1
	if mark:InvokeServer("batch", { remote = r:GetFullName() }) == "clean" then
		continue
	end
	for _, spec in ipairs(args) do
		if os.clock() - t0 > budget then
			table.insert(skipped, r:GetFullName() .. " " .. spec[1])
			continue
		end
		local a = spec[2]()
		if a == nil then continue end
		if spec[3] then a.n = 1 end
		local _, err = fire(r, a)
		local ret2, err2 = fire(r, a)
		local shown = ret2
		if typeof(shown) == "Instance" then shown = shown:GetFullName() end
		if type(shown) == "table" then shown = "(table)" end
		if type(shown) == "number" and shown ~= shown then shown = "NaN" end
		mark:InvokeServer("case", { remote = r:GetFullName(), class = r.ClassName, arg = spec[1],
			returned = if shown ~= nil then tostring(shown) else nil, error = err or err2 })
	end
end
mark:InvokeServer("done", { skipped = skipped, planned = planned, batches = cases })
end)
if not okRun then
	pcall(function() mark:InvokeServer("crash", why) end)
end
'''

# ---------------------------------------------------------------- shared by the templates: the data-store seam
SEAM = r'''local function dataStores()
	-- In Studio only, MifBridge's DataStore stand-in (rbx_play_probe fault_store) answers when it is there; a live
	-- server always gets DataStoreService.
	if game:GetService("RunService"):IsStudio() then
		local s = workspace:FindFirstChild("MifScratch")
		local m = s and s:FindFirstChild("DataStoreStandIn")
		if m and m:IsA("ModuleScript") then
			return require(m)
		end
	end
	return game:GetService("DataStoreService")
end
'''

# ---------------------------------------------------------------- template "save": the profile layer (ModuleScript)
SAVE_MODULE = r'''-- __NAME__Store: session-locked player saves. Written by MifBridge's rbx_script_scaffold (kind save).
-- Each load is a session with its own id, held in the stored record's Lock. A save lands only over its own lock and
-- answers false otherwise. Writes to one key go one at a time, in order. A profile whose save on leave failed is kept
-- on this server and retried; a rejoin here takes it. BindToClose waits for every save in flight; a leave runs once.
-- ProcessReceipt answers PurchaseGranted only after the save holding the grant landed.
-- A rewarded video ad pays through the same receipt handler (its reward is a developer product), so an ad's reward
-- lands under the same rule; the daily cap and the cooldown are counted in the profile and saved with the grant.
local Players = game:GetService("Players")
local HttpService = game:GetService("HttpService")

local Store = {}
Store.STORE_NAME = "__NAME__Profiles"
Store.DEFAULTS = { Coins = 0, Items = {} } -- Items: { [uid] = itemId }; change to your game's data
Store.LOCK_WAIT = 8       -- seconds a join waits for another server to release the record
Store.LOCK_STALE = 1800   -- a lock older than this belongs to a server that is gone, and is taken over
Store.LEAVE_TRIES = 4     -- tries of the save on leave before the profile is kept for the autosave to retry
Store.AUTOSAVE = 60
Store.RECEIPTS_KEPT = 100
Store.SESSION = HttpService:GenerateGUID(false)
Store.GRANTS = {}         -- [productId] = function(profile, receipt): change profile.Data; never yield
Store.isPresent = function(userId)
	return Players:GetPlayerByUserId(userId) ~= nil
end

-- Rewarded video ads: off until AD_PRODUCT is set and GRANTS holds that product's grant.
-- Read from Studio 0.748's API dump: AdService:GetAdAvailabilityNowAsync(adFormat) answers a Dictionary,
-- :CreateAdRewardFromDevProductId(devProductId) an AdReward, :ShowRewardedVideoAdAsync(player, reward, placementId?)
-- an Enum.ShowAdResult; Enum.AdFormat.RewardedVideo, Enum.AdAvailabilityResult.IsAvailable,
-- Enum.ShowAdResult.ShowCompleted and Enum.ProductPurchaseChannel.AdReward exist.
-- MEASURED on Studio 0.748 (ledger RB2947): GetAdAvailabilityNowAsync is a client call - from a server it fails "Local
-- player is not available", from a LocalScript it answers { AdAvailabilityResult = <an Enum.AdAvailabilityResult> }.
-- So the client reads it and sends the item's name with each ask (__NAME__AdClient), and the server takes only a name
-- that is an AdAvailabilityResult item; every other check (product, grant, profile, cap, cooldown) is the server's own,
-- and the reward comes only from Roblox's receipt, so a client that lies about availability gains nothing.
-- ASSUMED from Roblox's documentation, not measured (an ad cannot show in Studio): the receipt's field is
-- ProductPurchaseChannel; CreateAdRewardFromDevProductId and ShowRewardedVideoAdAsync are made from the server; the
-- reward's receipt reaches ProcessReceipt like a purchase's. If a receipt carries no channel field, one for
-- AD_PRODUCT that follows a show from this server is counted as the ad's. Check each on a published place.
Store.AD_PRODUCT = 0      -- the developer product id of the ad's reward; 0 leaves ads off
Store.AD_PLACEMENT = nil  -- an optional placement id for ShowRewardedVideoAdAsync
Store.AD_COOLDOWN = 300   -- seconds between one player's ads
Store.AD_DAILY_CAP = 5    -- ad rewards per player per UTC day
Store.ads = function() return game:GetService("AdService") end
Store.now = function() return os.time() end

__SEAM__
local ds = nil
local function store()
	ds = ds or dataStores():GetDataStore(Store.STORE_NAME)
	return ds
end

local profiles = {} -- key -> { Key, Data, Receipts, State = "active" | "closing", Leaving = bool }
local queues = {}   -- key -> { nextTicket, serving }

-- A test save slot: the creator's account owns every game pass of its own experience, so a first live
-- play on the real save, with every pass granted, cannot judge the game's pacing. A player allowed the slot (the
-- creator of a user-owned game, or TEST_SLOT_USERS) switches it with setTestSlot, which writes only its own choice key;
-- the choice is read once at the next join (chooseKey), and that session then uses a test key of its own - a new one
-- for each fresh start - so the real save is never read or written while it is on, and ownsPass answers false.
Store.TEST_SLOT_USERS = {} -- [userId] = true: may use the test slot besides the creator
Store.testSlotAllowed = function(userId)
	if Store.TEST_SLOT_USERS[userId] then return true end
	return game.CreatorType == Enum.CreatorType.User and userId > 0 and game.CreatorId == userId
end
Store.marketplace = function() return game:GetService("MarketplaceService") end
local active = {} -- [userId] = the key this session uses, when it is not the real one

function Store.realKey(userId)
	return "Player_" .. tostring(userId)
end

function Store.choiceKey(userId)
	return "TestSlotChoice_" .. tostring(userId)
end

function Store.keyFor(userId)
	return active[userId] or Store.realKey(userId)
end

function Store.testing(userId)
	return active[userId] ~= nil
end

-- At join: read the choice once; the answer is this session's key until forgetKey (at leave).
function Store.chooseKey(userId)
	active[userId] = nil
	if not Store.testSlotAllowed(userId) then return Store.realKey(userId) end
	local ok, c = pcall(function() return store():GetAsync(Store.choiceKey(userId)) end)
	if ok and type(c) == "table" and c.On == true and tonumber(c.Slot) then
		active[userId] = Store.realKey(userId) .. "_Test" .. tostring(math.floor(tonumber(c.Slot)))
	end
	return Store.keyFor(userId)
end

function Store.forgetKey(userId)
	active[userId] = nil
end

-- The switch, written to its own key only and applied at the next join; fresh true starts a new test save.
function Store.setTestSlot(userId, on, fresh)
	if not Store.testSlotAllowed(userId) then return false, "not allowed" end
	local ok, err = pcall(function()
		store():UpdateAsync(Store.choiceKey(userId), function(old)
			old = if type(old) == "table" then old else {}
			local slot = tonumber(old.Slot) or 0
			if fresh then slot += 1 end
			return { On = on == true, Slot = slot }
		end)
	end)
	if not ok then return false, tostring(err) end
	return true, "applies at the next join"
end

-- An owned pass, as Roblox answers it; false while the test slot is on, and false when the read fails.
function Store.ownsPass(userId, passId)
	if Store.testing(userId) then return false end
	local ok, owns = pcall(function() return Store.marketplace():UserOwnsGamePassAsync(userId, passId) end)
	return ok and owns == true
end

local function deepCopy(v)
	if type(v) ~= "table" then return v end
	local out = {}
	for k, x in pairs(v) do out[k] = deepCopy(x) end
	return out
end

local function reconcile(data)
	data = if type(data) == "table" then data else {}
	for k, v in pairs(Store.DEFAULTS) do
		if data[k] == nil then data[k] = deepCopy(v) end
	end
	return data
end

-- The ad count kept in the record: the UTC day it counts, how many rewards that day, and the last reward's time.
local function adState(saved)
	saved = if type(saved) == "table" then saved else {}
	return { Day = tonumber(saved.Day) or 0, Count = tonumber(saved.Count) or 0, Last = tonumber(saved.Last) or 0 }
end

-- One write at a time per key, first come first served.
local function queued(key, fn)
	local q = queues[key]
	if q == nil then
		q = { nextTicket = 1, serving = 1 }
		queues[key] = q
	end
	local ticket = q.nextTicket
	q.nextTicket += 1
	while q.serving ~= ticket do task.wait() end
	local ok, a, b = pcall(fn)
	q.serving += 1
	if not ok then error(a, 0) end
	return a, b
end

local function ours(lock)
	return type(lock) == "table" and lock.Session == Store.SESSION
end

local function foreign(lock)
	return type(lock) == "table" and lock.Session ~= Store.SESSION and os.time() - (tonumber(lock.Time) or 0) < Store.LOCK_STALE
end

-- Load: wait for a foreign lock up to LOCK_WAIT, then take the record under this session's lock.
function Store.load(key)
	local held = profiles[key]
	if held then
		if held.State == "closing" then
			-- kept after a failed save on leave: it is newer than the store, so this join takes it
			held.State = "active"
			return held
		end
		return nil, "already loaded on this server"
	end
	local deadline = os.clock() + Store.LOCK_WAIT
	local why = nil
	while true do
		local blocked = false
		local ok, rec = pcall(queued, key, function()
			return store():UpdateAsync(key, function(old)
				old = if type(old) == "table" then old else {}
				if foreign(old.Lock) then
					blocked = true
					return nil -- never write over another server's lock
				end
				old.Lock = { Session = Store.SESSION, Time = os.time() }
				return old
			end)
		end)
		if ok and rec and not blocked and ours(rec.Lock) then
			local p = { Key = key, Data = reconcile(rec.Data), Receipts = rec.Receipts or {}, State = "active", Leaving = false,
				Ads = adState(rec.Ads), AdShown = 0, AdShowing = false, AdPending = 0 }
			profiles[key] = p
			return p
		end
		why = if blocked then "locked by another server" elseif ok then "the record did not take this session's lock" else tostring(rec)
		if os.clock() >= deadline then return nil, why end
		task.wait(1)
	end
end

-- The active profile, or nil once its leave began (so nothing writes to a closing profile).
function Store.get(key)
	local p = profiles[key]
	return if p and p.State == "active" then p else nil
end

local function trimReceipts(p)
	local ids = {}
	for id, t in pairs(p.Receipts) do table.insert(ids, { id = id, t = t }) end
	if #ids <= Store.RECEIPTS_KEPT then return end
	table.sort(ids, function(a, b) return a.t < b.t end)
	for i = 1, #ids - Store.RECEIPTS_KEPT do p.Receipts[ids[i].id] = nil end
end

-- Write a profile. Lands only over this session's lock; release drops the lock when the profile is still closing.
-- Answers true when the write landed, else false and why.
local function write(p, release)
	local ok, landed, why = pcall(queued, p.Key, function()
		local wrote = false
		store():UpdateAsync(p.Key, function(old)
			old = if type(old) == "table" then old else {}
			if not ours(old.Lock) then
				wrote = false
				return nil
			end
			old.Data = deepCopy(p.Data)
			old.Receipts = deepCopy(p.Receipts)
			old.Ads = deepCopy(p.Ads)
			if release and p.State ~= "active" then
				old.Lock = nil
			else
				old.Lock.Time = os.time()
			end
			wrote = true
			return old
		end)
		return wrote, if wrote then nil else "the record is not under this session's lock"
	end)
	if not ok then return false, tostring(landed) end
	return landed, why
end

function Store.save(key)
	local p = profiles[key]
	if p == nil then return false, "no profile" end
	return write(p, false)
end

-- Leave: closed at once, then saved with the lock released, retried; kept on this server if every try failed.
-- A second caller waits for the first instead of saving again.
function Store.leave(key)
	local p = profiles[key]
	if p == nil then return true end
	if p.Leaving then
		while p.Leaving do task.wait() end
		return profiles[key] == nil
	end
	p.Leaving = true
	p.State = "closing"
	local landed = false
	for i = 1, Store.LEAVE_TRIES do
		landed = write(p, true)
		if landed or p.State == "active" or i == Store.LEAVE_TRIES then break end
		task.wait(i)
	end
	if landed and p.State == "closing" then
		profiles[key] = nil
	end
	p.Leaving = false
	return landed
end

-- BindToClose: every profile still held (active, closing or kept) is saved and released; waits for them all.
function Store.flush(deadline)
	local running = 0
	for key in pairs(profiles) do
		running += 1
		task.spawn(function()
			pcall(Store.leave, key)
			running -= 1
		end)
	end
	while running > 0 and os.clock() < deadline do task.wait() end
	return running == 0
end

local function adDay()
	return math.floor(Store.now() / 86400)
end

-- The client's availability read (GetAdAvailabilityNowAsync is a client call, RB2947), as the name it sent: the name
-- when it is an Enum.AdAvailabilityResult item, else nil.
local function availabilityName(report)
	if type(report) ~= "string" or #report > 40 then return nil end
	local ok, item = pcall(function() return Enum.AdAvailabilityResult[report] end) -- an unknown name errors
	if ok and typeof(item) == "EnumItem" then return item.Name end
	return nil
end

-- May this player be offered an ad now? report is the AdAvailabilityResult name the player's client read. true, or
-- false and why (and the seconds left of a cooldown). The server's own checks come first and decide; the client's read
-- only stops an ad it says cannot show. No profile, no ad: the count lives in the save and nowhere else.
function Store.adAvailable(player, report)
	if type(Store.AD_PRODUCT) ~= "number" or Store.AD_PRODUCT <= 0 then return false, "no ad product set" end
	if Store.GRANTS[Store.AD_PRODUCT] == nil then return false, "no grant for the ad product" end
	local p = Store.get(Store.keyFor(player.UserId))
	if p == nil then return false, "no profile" end
	if p.AdShowing then return false, "an ad is showing" end
	local today = if p.Ads.Day == adDay() then p.Ads.Count else 0
	if today >= Store.AD_DAILY_CAP then return false, "daily cap" end
	local left = Store.AD_COOLDOWN - (Store.now() - math.max(p.Ads.Last, p.AdShown))
	if left > 0 then return false, "cooldown", left end
	local name = availabilityName(report)
	if name == nil then return false, "availability read failed" end
	if name ~= Enum.AdAvailabilityResult.IsAvailable.Name then return false, "no ad available" end
	return true
end

-- Show one ad to a player, from the server, with the availability its client read. Answers true when it played to
-- the end, else false and why. It pays nothing: the reward arrives as a receipt and processReceipt grants it.
function Store.showAd(player, report)
	local ok, why, left = Store.adAvailable(player, report)
	if not ok then return false, why, left end
	local p = Store.get(Store.keyFor(player.UserId))
	if p == nil then return false, "no profile" end
	if p.AdShowing then return false, "an ad is showing" end -- the availability read yields
	p.AdShowing = true
	p.AdPending += 1
	local shown, result = pcall(function()
		local ads = Store.ads()
		return ads:ShowRewardedVideoAdAsync(player, ads:CreateAdRewardFromDevProductId(Store.AD_PRODUCT), Store.AD_PLACEMENT)
	end)
	p.AdShowing = false
	if not shown or result ~= Enum.ShowAdResult.ShowCompleted then
		p.AdPending = math.max(0, p.AdPending - 1)
		return false, if shown and typeof(result) == "EnumItem" then result.Name else "the show failed"
	end
	p.AdShown = Store.now() -- the cooldown runs from here until the receipt sets the saved time
	return true
end

-- Count a receipt that is an ad's reward: once, beside its grant, so the count is in the save that holds the grant.
local function countAd(p, receipt)
	local channel = receipt.ProductPurchaseChannel
	local isAd = channel == Enum.ProductPurchaseChannel.AdReward
		or (channel == nil and receipt.ProductId == Store.AD_PRODUCT and p.AdPending > 0)
	if not isAd then return end
	local day = adDay()
	if p.Ads.Day ~= day then
		p.Ads.Day = day
		p.Ads.Count = 0
	end
	p.Ads.Count += 1
	p.Ads.Last = Store.now()
	p.AdPending = math.max(0, p.AdPending - 1)
end

-- ProcessReceipt: the grant and its receipt record go into the profile together; PurchaseGranted only after the save
-- holding them landed. A failed save keeps both and answers NotProcessedYet; Roblox's retry finds the record and only
-- saves. An absent player, or a product with no grant, is NotProcessedYet. An ad's reward is one more receipt here.
function Store.processReceipt(receipt)
	local NotYet = Enum.ProductPurchaseDecision.NotProcessedYet
	if not Store.isPresent(receipt.PlayerId) then return NotYet end
	local key = Store.keyFor(receipt.PlayerId)
	local p = Store.get(key)
	if p == nil then return NotYet end
	local id = tostring(receipt.PurchaseId)
	if p.Receipts[id] == nil then
		local grant = Store.GRANTS[receipt.ProductId]
		if grant == nil then
			warn("__NAME__Store: no grant for product " .. tostring(receipt.ProductId))
			return NotYet
		end
		grant(p, receipt)
		p.Receipts[id] = os.time()
		countAd(p, receipt)
		trimReceipts(p)
	end
	if write(p, false) then
		return Enum.ProductPurchaseDecision.PurchaseGranted
	end
	return NotYet
end

-- Every item uid held by each profile on this server (active and kept); for a census.
function Store.census()
	local out = {}
	for key, p in pairs(profiles) do
		local uids = {}
		for uid in pairs(p.Data.Items or {}) do table.insert(uids, tostring(uid)) end
		out[key] = uids
	end
	return out
end

-- The autosave, and the retry of kept profiles.
function Store.autosaveOnce()
	for key, p in pairs(profiles) do
		if p.State == "active" then
			write(p, false)
		elseif not p.Leaving then
			task.spawn(Store.leave, key)
		end
	end
end

return Store
'''

SAVE_SERVER = r'''-- __NAME__Server: wires __NAME__Store to players, receipts and shutdown. Written by MifBridge's rbx_script_scaffold.
local Players = game:GetService("Players")
local MarketplaceService = game:GetService("MarketplaceService")
local Store = require(game:GetService("ServerScriptService"):WaitForChild("__NAME__Store"))

local function join(player)
	local key = Store.chooseKey(player.UserId) -- the test slot's key when its owner switched it on before this join
	local p, why = Store.load(key)
	if p == nil then
		if player.Parent then
			player:Kick("Your save could not be loaded (" .. tostring(why) .. "). Please rejoin in a minute.")
		end
		return
	end
	if player.Parent == nil then
		Store.leave(key) -- left while loading
	end
end

Players.PlayerAdded:Connect(join)
for _, player in ipairs(Players:GetPlayers()) do
	task.spawn(join, player)
end
Players.PlayerRemoving:Connect(function(player)
	Store.leave(Store.keyFor(player.UserId))
	Store.forgetKey(player.UserId)
end)
game:BindToClose(function()
	Store.flush(os.clock() + 25)
end)
MarketplaceService.ProcessReceipt = Store.processReceipt
task.spawn(function()
	while true do
		task.wait(Store.AUTOSAVE)
		Store.autosaveOnce()
	end
end)

-- Rewarded ads: the client reads the ad's availability (a client call, RB2947) and sends it with each ask, whether to
-- show its button and for the ad; the server decides both with its own checks.
local adRemote = game:GetService("ReplicatedStorage"):WaitForChild("__NAME__AdRemote")
local adAsked = {} -- [player] = os.clock() of the last ask: one ask per player every 2 seconds
Players.PlayerRemoving:Connect(function(player)
	adAsked[player] = nil
end)
adRemote.OnServerInvoke = function(player, what, report)
	if what ~= "available" and what ~= "show" then return false, "not a request" end
	local t = os.clock()
	if adAsked[player] and t - adAsked[player] < 2 then return false, "too soon" end
	adAsked[player] = t
	if what == "show" then
		return Store.showAd(player, report)
	end
	return Store.adAvailable(player, report)
end
'''

SAVE_AD_CLIENT = r'''-- __NAME__AdClient: a "watch an ad" button, shown only while the server says an ad can be offered to this player.
-- Written by MifBridge's rbx_script_scaffold (kind save). The client reads the ad's availability (AdService:
-- GetAdAvailabilityNowAsync answers only a client: ledger RB2947) and sends its AdAvailabilityResult name with each
-- ask; __NAME__Store on the server checks the product, the daily cap and the cooldown, shows the ad, and grants the
-- reward from its receipt. With Store.AD_PRODUCT unset the server answers false and no button shows. Restyle the
-- button; keep the two calls.
local Players = game:GetService("Players")
local AdService = game:GetService("AdService")
local remote = game:GetService("ReplicatedStorage"):WaitForChild("__NAME__AdRemote")

-- The availability as an Enum.AdAvailabilityResult item's name, or nil when the read failed.
local function availability()
	local ok, res = pcall(function()
		return AdService:GetAdAvailabilityNowAsync(Enum.AdFormat.RewardedVideo)
	end)
	if ok and type(res) == "table" and typeof(res.AdAvailabilityResult) == "EnumItem" then
		return res.AdAvailabilityResult.Name
	end
	return nil
end

local gui = Instance.new("ScreenGui")
gui.Name = "__NAME__AdGui"
gui.ResetOnSpawn = false
gui.Enabled = false
local button = Instance.new("TextButton")
button.Name = "Watch"
button.AnchorPoint = Vector2.new(1, 1)
button.Position = UDim2.new(1, -16, 1, -16)
button.Size = UDim2.fromOffset(220, 44)
button.Text = "Watch an ad for a reward"
button.Parent = gui
gui.Parent = Players.LocalPlayer:WaitForChild("PlayerGui")

local busy = false
local function refresh()
	local ok, available = pcall(function() return remote:InvokeServer("available", availability()) end)
	gui.Enabled = ok and available == true
end
button.Activated:Connect(function()
	if busy then return end
	busy = true
	gui.Enabled = false
	pcall(function() return remote:InvokeServer("show", availability()) end)
	task.wait(3)
	busy = false
	refresh()
end)
while true do
	if not busy then refresh() end
	task.wait(30)
end
'''

# Self-test of the save template: fake keys, driven through the module, against the stand-in. Runs only in Studio when
# Workspace.MifScratch.SelfTest___NAME__ and the stand-in are there (rbx_play_probe self_test puts both in, then removes them).
SAVE_SELFTEST = r'''-- __NAME__SaveSelfTest: the save template's self-test. Runs only in Studio, only while MifBridge's play probe has put
-- Workspace.MifScratch.SelfTest___NAME__ and the DataStore stand-in there; otherwise it does nothing.
local RunService = game:GetService("RunService")
local HttpService = game:GetService("HttpService")
if not RunService:IsStudio() then return end
local scratch = workspace:WaitForChild("MifScratch", 5)
if scratch == nil then return end
if scratch:WaitForChild("SelfTest___NAME__", 20) == nil then return end
local S = require(scratch:WaitForChild("DataStoreStandIn"))
local Store = require(game:GetService("ServerScriptService"):WaitForChild("__NAME__Store"))
local NAME = Store.STORE_NAME
local cases = {}
local function case(name, fn)
	local ok, pass, got = pcall(fn)
	table.insert(cases, { name = name, pass = ok and pass == true, got = if ok then got else "error: " .. tostring(pass) })
end
local function failNext(key, n, op)
	S.schedule({ op = op or "UpdateAsync", store = NAME, key = key, nth = 1, count = n or 1, ["do"] = "fail",
		after = 0 })
end
-- each fail event counts calls from when it is added: schedule uses matching calls, so add with a fresh key filter
local A, B, C, D = "SelfTest_A", "SelfTest_B", "SelfTest_C", "SelfTest_D"
Store.LOCK_WAIT = 3
Store.LEAVE_TRIES = 2

case("a save over another server's lock answers false and writes nothing", function()
	local p = assert(Store.load(D))
	p.Data.Coins = 5
	assert(Store.save(D))
	S.setLock(NAME, D, "OtherServer")
	p.Data.Coins = 999
	local landed = Store.save(D)
	local rec = S.peek(NAME, D)
	return landed == false and rec.Data.Coins == 5, { landed = landed, stored = rec.Data.Coins }
end)

case("a join waits for a foreign lock, then loads when it is released", function()
	S.setLock(NAME, C, "OtherServer")
	task.delay(1.5, function() S.setLock(NAME, C, nil) end)
	local t = os.clock()
	local p, why = Store.load(C)
	local waited = os.clock() - t
	return p ~= nil and waited >= 1.4, { loaded = p ~= nil, waited = waited, why = why }
end)

case("a join under a lock never released gives up after LOCK_WAIT without writing", function()
	local k = "SelfTest_E"
	S.setLock(NAME, k, "OtherServer")
	local p, why = Store.load(k)
	local rec = S.peek(NAME, k)
	return p == nil and rec.Lock.Session == "OtherServer", { loaded = p ~= nil, why = why }
end)

case("a failed save on leave keeps the profile; a rejoin takes it, so a traded item never comes back", function()
	local a = assert(Store.load(A))
	local b = assert(Store.load(B))
	a.Data.Items["uid-1"] = "Sword"
	assert(Store.save(A))
	a.Data.Items["uid-1"] = nil -- traded to B
	b.Data.Items["uid-1"] = "Sword"
	assert(Store.save(B))
	failNext(A, Store.LEAVE_TRIES)
	local landed = Store.leave(A)
	local back = Store.load(A)
	local census = Store.census()
	local count = 0
	for _, uids in pairs(census) do
		for _, u in ipairs(uids) do
			if u == "uid-1" then count += 1 end
		end
	end
	return landed == false and back ~= nil and count == 1, { leaveLanded = landed, rejoined = back ~= nil, held = count }
end)

case("writes to one key never overlap: a stalled save and the next run one after the other", function()
	local p = Store.get(A)
	S.schedule({ op = "UpdateAsync", store = NAME, key = A, nth = 1, count = 1, ["do"] = "stall", seconds = 1, after = 0 })
	p.Data.Coins = 1
	local done = 0
	task.spawn(function() Store.save(A); done += 1 end)
	task.wait()
	p.Data.Coins = 2
	task.spawn(function() Store.save(A); done += 1 end)
	local t = os.clock()
	while done < 2 and os.clock() - t < 10 do task.wait() end
	local rec = S.peek(NAME, A)
	return S.maxConcurrent(NAME, A) == 1 and rec.Data.Coins == 2, { maxConcurrent = S.maxConcurrent(NAME, A), stored = rec.Data.Coins }
end)

case("a leave called twice saves once", function()
	local before = S.callsFor(NAME, A, "UpdateAsync")
	local r1, r2
	task.spawn(function() r1 = Store.leave(A) end)
	task.spawn(function() r2 = Store.leave(A) end)
	local t = os.clock()
	while (r1 == nil or r2 == nil) and os.clock() - t < 10 do task.wait() end
	local writes = S.callsFor(NAME, A, "UpdateAsync") - before
	local rec = S.peek(NAME, A)
	return r1 == true and r2 == true and writes == 1 and rec.Lock == nil, { writes = writes, released = rec.Lock == nil }
end)

case("a receipt whose save failed is granted once across Roblox's retry", function()
	local key = Store.keyFor(-9001)
	local p = assert(Store.load(key))
	local present = Store.isPresent
	Store.isPresent = function(id) return id == -9001 end
	Store.GRANTS[424242] = function(prof) prof.Data.Coins += 100 end
	local receipt = { PlayerId = -9001, PurchaseId = "R1", ProductId = 424242 }
	failNext(key, 1)
	local first = Store.processReceipt(receipt)
	local second = Store.processReceipt(receipt)
	local rec = S.peek(NAME, key)
	Store.isPresent = present
	local granted = rec and rec.Data and rec.Data.Coins
	return first == Enum.ProductPurchaseDecision.NotProcessedYet and second == Enum.ProductPurchaseDecision.PurchaseGranted
		and p.Data.Coins == 100 and granted == 100, { first = first.Name, second = second.Name, coins = p.Data.Coins, stored = granted }
end)

case("PurchaseGranted comes only after the save holding the grant landed", function()
	local key = Store.keyFor(-9002)
	assert(Store.load(key))
	local present = Store.isPresent
	Store.isPresent = function(id) return id == -9002 end
	Store.GRANTS[424243] = function(prof) prof.Data.Coins += 1 end
	S.schedule({ op = "UpdateAsync", store = NAME, key = key, nth = 1, count = 1, ["do"] = "stall", seconds = 1, after = 0 })
	local answer = Store.processReceipt({ PlayerId = -9002, PurchaseId = "R2", ProductId = 424243 })
	local rec = S.peek(NAME, key)
	Store.isPresent = present
	return answer == Enum.ProductPurchaseDecision.PurchaseGranted and rec.Receipts.R2 ~= nil, { answer = answer.Name, recorded = rec.Receipts.R2 ~= nil }
end)

case("a receipt for a player who is not there is NotProcessedYet", function()
	local answer = Store.processReceipt({ PlayerId = -9003, PurchaseId = "R3", ProductId = 424242 })
	return answer == Enum.ProductPurchaseDecision.NotProcessedYet, answer.Name
end)

-- Rewarded ads: a recorder stands in for the ad service (an ad cannot show in Studio), on a fake player.
local function adRecorder(available, result)
	local rec = { asked = 0, made = {}, shown = {} }
	function rec:GetAdAvailabilityNowAsync(format)
		rec.asked += 1
		if available == "error" then error("recorder: the availability read failed") end
		return { AdAvailabilityResult = available }
	end
	function rec:CreateAdRewardFromDevProductId(id)
		table.insert(rec.made, id)
		return { product = id }
	end
	function rec:ShowRewardedVideoAdAsync(player, reward, placement)
		table.insert(rec.shown, { user = player.UserId, product = reward.product })
		return result
	end
	return rec
end
do
	local AD, USER = 424244, -9004
	local Available, Completed = Enum.AdAvailabilityResult.IsAvailable, Enum.ShowAdResult.ShowCompleted
	local OK = Available.Name -- what the client sends when its availability read says an ad can show
	local Granted, NotYet = Enum.ProductPurchaseDecision.PurchaseGranted, Enum.ProductPurchaseDecision.NotProcessedYet
	local key = Store.keyFor(USER)
	local player = { UserId = USER }
	local keep = { ads = Store.ads, now = Store.now, product = Store.AD_PRODUCT, cooldown = Store.AD_COOLDOWN,
		cap = Store.AD_DAILY_CAP, present = Store.isPresent }
	local rec = adRecorder(Available, Completed)
	local function use(r)
		rec = r
		Store.ads = function() return rec end
	end
	use(rec)
	Store.isPresent = function(id) return id == USER end
	Store.GRANTS[AD] = function(prof) prof.Data.Coins += 10 end
	local function adReceipt(id, channel)
		return { PlayerId = USER, PurchaseId = id, ProductId = AD, ProductPurchaseChannel = channel }
	end

	case("ads are off until the ad product is set: nothing is asked of the ad service", function()
		assert(Store.load(key))
		Store.AD_PRODUCT = 0
		local ok, why = Store.adAvailable(player, OK)
		local shown = Store.showAd(player, OK)
		Store.AD_PRODUCT = AD
		return ok == false and why == "no ad product set" and shown == false and rec.asked == 0 and #rec.shown == 0,
			{ available = ok, why = why, asked = rec.asked, shown = #rec.shown }
	end)

	case("an ad reward whose save failed is granted once and counted once across Roblox's retry", function()
		local p = Store.get(key)
		local shown, why = Store.showAd(player, OK)
		failNext(key, 1)
		local receipt = adReceipt("AD1", Enum.ProductPurchaseChannel.AdReward)
		local first = Store.processReceipt(receipt)
		local before = S.peek(NAME, key)
		local storedBefore = (before and before.Data and before.Data.Coins) or 0
		local second = Store.processReceipt(receipt)
		local after = S.peek(NAME, key)
		return shown == true and #rec.shown == 1 and rec.shown[1].product == AD and first == NotYet and storedBefore == 0
			and second == Granted and p.Data.Coins == 10 and after.Data.Coins == 10 and after.Ads.Count == 1
			and after.Receipts.AD1 ~= nil,
			{ shown = shown, why = why, first = first.Name, second = second.Name, storedBefore = storedBefore,
				stored = after.Data.Coins, counted = after.Ads and after.Ads.Count }
	end)

	case("the cooldown refuses the next ad without asking the ad service", function()
		local asked = rec.asked
		local ok, why, left = Store.adAvailable(player, OK)
		local shown = Store.showAd(player, OK)
		return ok == false and why == "cooldown" and type(left) == "number" and left > 0 and shown == false
			and rec.asked == asked and #rec.shown == 1, { why = why, left = left, asked = rec.asked - asked, shown = #rec.shown }
	end)

	case("the daily cap refuses an ad, is read back from the save on a rejoin, and opens on a new day", function()
		Store.AD_COOLDOWN = 0
		Store.AD_DAILY_CAP = 2
		local shown = Store.showAd(player, OK)
		-- no channel field on this one: a receipt for the ad product after this server's show still counts
		local answer = Store.processReceipt(adReceipt("AD2", nil))
		local asked = rec.asked
		local ok, why = Store.adAvailable(player, OK)
		local left = Store.leave(key)
		local back = assert(Store.load(key))
		local ok2, why2 = Store.adAvailable(player, OK)
		local askedAtCap = rec.asked - asked
		local real = Store.now
		Store.now = function() return real() + 86400 end
		local ok3 = Store.adAvailable(player, OK)
		Store.now = real
		return shown == true and answer == Granted and ok == false and why == "daily cap" and left == true
			and back.Ads.Count == 2 and back.Data.Coins == 20 and ok2 == false and why2 == "daily cap" and askedAtCap == 0
			and ok3 == true, { answer = answer.Name, why = why, saved = back.Ads.Count, afterRejoin = why2,
				askedAtCap = askedAtCap, nextDay = ok3 }
	end)

	case("no ad is shown when none is available or the availability read fails", function()
		Store.AD_DAILY_CAP = 10
		-- the client's read (RB2947: GetAdAvailabilityNowAsync answers only a client) said NoFill, then failed
		use(adRecorder(Available, Completed))
		local a, whyA = Store.showAd(player, Enum.AdAvailabilityResult.NoFill.Name)
		local none = #rec.shown
		local b, whyB = Store.showAd(player, nil)
		local c, whyC = Store.showAd(player, 42)
		return a == false and whyA == "no ad available" and b == false and whyB == "availability read failed"
			and c == false and whyC == "availability read failed" and none == 0 and #rec.shown == 0 and rec.asked == 0,
			{ noFill = whyA, failed = whyB, notAName = whyC, asked = rec.asked }
	end)

	case("an ad that did not complete counts nothing and starts no cooldown", function()
		local p = Store.get(key)
		Store.AD_COOLDOWN = 300
		p.Ads.Last, p.AdShown = 0, 0
		use(adRecorder(Available, Enum.ShowAdResult.ShowInterrupted))
		local shown, why = Store.showAd(player, OK)
		local again = Store.adAvailable(player, OK)
		return shown == false and why == "ShowInterrupted" and #rec.shown == 1 and p.Ads.Count == 2 and p.AdPending == 0
			and p.AdShown == 0 and again == true, { why = why, counted = p.Ads.Count, pending = p.AdPending, again = again }
	end)

	case("a purchase of the ad product through the shop is granted and is not counted as an ad", function()
		local p = Store.get(key)
		local answer = Store.processReceipt(adReceipt("BUY1", Enum.ProductPurchaseChannel.InExperience))
		local stored = S.peek(NAME, key)
		return answer == Granted and p.Data.Coins == 30 and stored.Data.Coins == 30 and stored.Ads.Count == 2,
			{ answer = answer.Name, coins = stored.Data.Coins, counted = stored.Ads.Count }
	end)

	Store.ads, Store.now = keep.ads, keep.now
	Store.AD_PRODUCT, Store.AD_COOLDOWN, Store.AD_DAILY_CAP, Store.isPresent = keep.product, keep.cooldown, keep.cap, keep.present
	Store.GRANTS[AD] = nil
end

-- the test save slot: the creator's account owns every pass of its own game, so it tests on a save of its own
do
	local USER, OTHER = -9010, -9011
	local real = Store.realKey(USER)
	Store.TEST_SLOT_USERS[USER] = true
	local keepMarket = Store.marketplace
	Store.marketplace = function() return { UserOwnsGamePassAsync = function() return true end } end
	local function stored(k) return HttpService:JSONEncode(S.peek(NAME, k) or {}) end
	case("the test slot is off by default: the real key, and owned passes as Roblox answers them", function()
		local k = Store.chooseKey(USER)
		return k == real and not Store.testing(USER) and Store.ownsPass(USER, 1) == true, { key = k }
	end)
	local p = assert(Store.load(real))
	p.Data.Coins = 7
	assert(Store.save(real))
	Store.leave(real)
	local before = stored(real)
	case("switching it on writes only its own key: this session keeps the real save", function()
		local ok, why = Store.setTestSlot(USER, true, true)
		return ok and Store.keyFor(USER) == real and stored(real) == before, { ok = ok, why = why }
	end)
	local firstTest = nil
	case("at the next join the session uses a test key, owned passes read false, and the real save is unchanged", function()
		local k = Store.chooseKey(USER)
		firstTest = k
		local tp = assert(Store.load(k))
		tp.Data.Coins = 999
		assert(Store.save(k))
		Store.leave(k)
		return k ~= real and Store.testing(USER) and Store.ownsPass(USER, 1) == false and stored(real) == before
			and S.peek(NAME, k).Data.Coins == 999, { key = k, realUnchanged = stored(real) == before }
	end)
	case("a fresh start is a new test key; switching off goes back to the real save as it was", function()
		Store.setTestSlot(USER, true, true)
		local second = Store.chooseKey(USER)
		Store.setTestSlot(USER, false)
		local back = Store.chooseKey(USER)
		local p2 = assert(Store.load(back))
		local coins = p2.Data.Coins
		Store.leave(back)
		return second ~= firstTest and second ~= real and back == real and coins == 7 and not Store.testing(USER),
			{ first = firstTest, second = second, back = back, coins = coins }
	end)
	case("a player not allowed the slot cannot switch it and stays on the real save", function()
		local ok = Store.setTestSlot(OTHER, true, true)
		return ok == false and Store.chooseKey(OTHER) == Store.realKey(OTHER), { ok = ok }
	end)
	Store.marketplace = keepMarket
	Store.TEST_SLOT_USERS[USER] = nil
	Store.forgetKey(USER)
end

-- last: flush releases every profile this server holds, the real players' too
case("flush waits for a save on leave still in flight", function()
	S.schedule({ op = "UpdateAsync", store = NAME, key = B, nth = 1, count = 1, ["do"] = "stall", seconds = 1.5, after = 0 })
	task.spawn(Store.leave, B)
	task.wait()
	local t = os.clock()
	local all = Store.flush(os.clock() + 10)
	local rec = S.peek(NAME, B)
	return all and rec.Lock == nil and os.clock() - t >= 1.2, { waited = os.clock() - t, released = rec.Lock == nil }
end)
-- hold A and B again, loaded from the store, so the probe's census reads what a rejoin would get
Store.load(A)
Store.load(B)

local passed = 0
for _, c in ipairs(cases) do
	if c.pass then passed += 1 end
end
local j = HttpService:JSONEncode({ template = "save", name = "__NAME__", passed = passed, total = #cases, cases = cases })
--@emit MIFSELFTEST __NAME__.save
'''

# ---------------------------------------------------------------- template "action": a server-checked action
ACTION_MODULE = r'''-- __NAME__Action: a server-checked action. Written by MifBridge's rbx_script_scaffold (kind action).
-- The client names its target by id; the server checks the id names a live target, the character's root is within
-- reach plus a lag allowance of the target's surface, and the cooldown. A swing at nothing, a dead target, one out of
-- reach or a forged id pays nothing and still spends the cooldown, so naming targets one after another never acts faster.
local CollectionService = game:GetService("CollectionService")
local HttpService = game:GetService("HttpService")

local Action = {}
Action.TAG = "__NAME__Target"        -- tag a BasePart or Model as a target; give it a Health attribute
Action.REACH = 7                     -- studs from the root to the target's surface
Action.LAG = 4                       -- allowance for the client being behind the server
Action.COOLDOWN = 0.5
Action.DAMAGE = 1
Action.now = os.clock                -- the clock; a self-test may replace it
-- Line of sight through a cell grid (a block game whose blocks exist as parts only where they touch air): set GRID to a
-- table with solid(x, y, z) -> true for a solid cell, cells being CELL studs on a side (GRID.CELL, else Action.CELL),
-- cell (x, y, z) holding the world points from (x, y, z) * CELL. The server walks the line from the head to just inside
-- the target's box in STEP stud steps, and a swing that meets a solid cell other than the target's own is refused.
-- nil (the default) checks no line.
Action.GRID = nil
Action.CELL = 4
Action.STEP = 0.2
Action.pay = function(player, target)
	-- what an accepted action gives; change to your game's reward
	player:SetAttribute("__NAME__Score", (player:GetAttribute("__NAME__Score") or 0) + 1)
end

local byId = {}
local last = {} -- player -> time of the last action, accepted or not

local function register(target)
	local id = target:GetAttribute("__NAME__Id")
	if type(id) ~= "string" then
		id = HttpService:GenerateGUID(false)
		target:SetAttribute("__NAME__Id", id)
	end
	byId[id] = target
end
for _, t in ipairs(CollectionService:GetTagged(Action.TAG)) do register(t) end
CollectionService:GetInstanceAddedSignal(Action.TAG):Connect(register)
CollectionService:GetInstanceRemovedSignal(Action.TAG):Connect(function(t)
	local id = t:GetAttribute("__NAME__Id")
	if id then byId[id] = nil end
end)

local function box(target)
	if target:IsA("BasePart") then return target.CFrame, target.Size end
	if target:IsA("Model") then return target:GetBoundingBox() end
	return nil
end

-- distance from a point to a box's surface (0 inside)
local function surfaceDistance(point, cf, size)
	local p = cf:PointToObjectSpace(point)
	local h = size / 2
	local dx = math.max(math.abs(p.X) - h.X, 0)
	local dy = math.max(math.abs(p.Y) - h.Y, 0)
	local dz = math.max(math.abs(p.Z) - h.Z, 0)
	return math.sqrt(dx * dx + dy * dy + dz * dz)
end

local function cellOf(p, cell)
	return math.floor(p.X / cell), math.floor(p.Y / cell), math.floor(p.Z / cell)
end

-- true when the line from the head to just inside the target's box meets no other solid cell; false, why when it does
local function clearLine(from, cf, size)
	local grid = Action.GRID
	if grid == nil then return true end
	local cell = (type(grid.CELL) == "number" and grid.CELL > 0) and grid.CELL or Action.CELL
	local local0 = cf:PointToObjectSpace(from)
	local h = size / 2
	-- the box's point nearest the head, moved a little inside towards its centre: where the swing lands
	local near = Vector3.new(math.clamp(local0.X, -h.X, h.X), math.clamp(local0.Y, -h.Y, h.Y), math.clamp(local0.Z, -h.Z, h.Z))
	local goal = cf:PointToWorldSpace(near * 0.9)
	local tx, ty, tz = cellOf(cf.Position, cell)
	local hx, hy, hz = cellOf(from, cell)
	local d = goal - from
	local n = math.max(1, math.ceil(d.Magnitude / Action.STEP))
	for i = 1, n do
		local p = from + d * (i / n)
		local x, y, z = cellOf(p, cell)
		if not (x == tx and y == ty and z == tz) and not (x == hx and y == hy and z == hz) then
			local ok, solid = pcall(grid.solid, x, y, z)
			if not ok then return false, "line of sight unread" end
			if solid then return false, "no line of sight" end
		end
	end
	return true
end

-- Answers accepted (bool) and why.
function Action.handle(player, id)
	local t = Action.now()
	if last[player] and t - last[player] < Action.COOLDOWN then
		return false, "cooldown"
	end
	last[player] = t -- spent from here on, whatever the outcome
	if type(id) ~= "string" or #id > 64 then return false, "bad id" end
	local target = byId[id]
	if target == nil or not target:IsDescendantOf(workspace) then return false, "no such target" end
	local hp = target:GetAttribute("Health")
	if type(hp) ~= "number" or hp ~= hp or hp <= 0 then return false, "dead" end
	local char = player.Character
	local root = char and char:FindFirstChild("HumanoidRootPart")
	if root == nil then return false, "no character" end
	local cf, size = box(target)
	if cf == nil then return false, "target has no box" end
	if surfaceDistance(root.Position, cf, size) > Action.REACH + Action.LAG then return false, "out of reach" end
	local head = char:FindFirstChild("Head")
	local sight, blocked = clearLine(if head and head:IsA("BasePart") then head.Position else root.Position, cf, size)
	if not sight then return false, blocked end
	target:SetAttribute("Health", hp - Action.DAMAGE)
	Action.pay(player, target)
	return true, "hit"
end

function Action.forget(player)
	last[player] = nil
end

return Action
'''

ACTION_SERVER = r'''-- __NAME__ActionServer: connects the remote to __NAME__Action. Written by MifBridge's rbx_script_scaffold.
local RS = game:GetService("ReplicatedStorage")
local Players = game:GetService("Players")
local Action = require(game:GetService("ServerScriptService"):WaitForChild("__NAME__Action"))
local remote = RS:WaitForChild("__NAME__Remote")
remote:SetAttribute("Cooldown", Action.COOLDOWN)
remote.OnServerEvent:Connect(function(player, id)
	Action.handle(player, id)
end)
Players.PlayerRemoving:Connect(Action.forget)
'''

ACTION_CLIENT = r'''-- __NAME__ActionClient: one hold loop fed by every input - mouse, the E key, the gamepad's right trigger and any
-- GuiButton tagged __NAME__Button - firing at the server's cooldown. Space is jump, so it is not used.
-- Written by MifBridge's rbx_script_scaffold.
local RS = game:GetService("ReplicatedStorage")
local Players = game:GetService("Players")
local CAS = game:GetService("ContextActionService")
local CollectionService = game:GetService("CollectionService")
local remote = RS:WaitForChild("__NAME__Remote")
local player = Players.LocalPlayer
local held = {} -- input name -> true while held

local function nearestTarget()
	local char = player.Character
	local root = char and char:FindFirstChild("HumanoidRootPart")
	if root == nil then return nil end
	local best, bestD = nil, math.huge
	for _, t in ipairs(CollectionService:GetTagged("__NAME__Target")) do
		local pos = if t:IsA("Model") then t:GetPivot().Position elseif t:IsA("BasePart") then t.Position else nil
		local id = t:GetAttribute("__NAME__Id")
		if pos and type(id) == "string" and (t:GetAttribute("Health") or 0) > 0 then
			local d = (pos - root.Position).Magnitude
			if d < bestD then best, bestD = id, d end
		end
	end
	return best
end

local looping = false
local function loop()
	if looping then return end
	looping = true
	while next(held) ~= nil do
		remote:FireServer(nearestTarget())
		task.wait(remote:GetAttribute("Cooldown") or 0.5)
	end
	looping = false
end

local function onAction(name, state)
	if state == Enum.UserInputState.Begin then
		held[name] = true
		task.spawn(loop)
	elseif state == Enum.UserInputState.End or state == Enum.UserInputState.Cancel then
		held[name] = nil
	end
	return Enum.ContextActionResult.Pass
end
-- low priority, and passing the input on, so menus that hold the selection still get it
CAS:BindActionAtPriority("__NAME__Act", onAction, false, Enum.ContextActionPriority.Low.Value,
	Enum.UserInputType.MouseButton1, Enum.KeyCode.E, Enum.KeyCode.ButtonR2)

local function hookButton(b)
	if not b:IsA("GuiButton") then return end
	b.MouseButton1Down:Connect(function() onAction("button:" .. b:GetFullName(), Enum.UserInputState.Begin) end)
	b.MouseButton1Up:Connect(function() onAction("button:" .. b:GetFullName(), Enum.UserInputState.End) end)
	b.MouseLeave:Connect(function() onAction("button:" .. b:GetFullName(), Enum.UserInputState.End) end)
end
for _, b in ipairs(CollectionService:GetTagged("__NAME__Button")) do hookButton(b) end
CollectionService:GetInstanceAddedSignal("__NAME__Button"):Connect(hookButton)
'''

ACTION_SELFTEST = r'''-- __NAME__ActionSelfTest: the action template's self-test. Runs only in Studio, only while MifBridge's play probe
-- has put Workspace.MifScratch.SelfTest___NAME__ there; it needs one player in the test.
local RunService = game:GetService("RunService")
local HttpService = game:GetService("HttpService")
local Players = game:GetService("Players")
local CollectionService = game:GetService("CollectionService")
if not RunService:IsStudio() then return end
local scratch = workspace:WaitForChild("MifScratch", 5)
if scratch == nil or scratch:WaitForChild("SelfTest___NAME__", 20) == nil then return end
local Action = require(game:GetService("ServerScriptService"):WaitForChild("__NAME__Action"))
local cases = {}
local function report()
	local passed = 0
	for _, c in ipairs(cases) do if c.pass then passed += 1 end end
	local j = HttpService:JSONEncode({ template = "action", name = "__NAME__", passed = passed, total = #cases, cases = cases })
	--@emit MIFSELFTEST __NAME__.action
end
local t0 = os.clock()
while #Players:GetPlayers() < 1 and os.clock() - t0 < 20 do task.wait(0.2) end
local player = Players:GetPlayers()[1]
local root = player and (player.Character or player.CharacterAdded:Wait()):WaitForChild("HumanoidRootPart", 10)
task.wait(2) -- let the character land
if root == nil then
	table.insert(cases, { name = "a player with a character joined", pass = false, got = "no character in 20 s" })
	report()
	return
end
local function target(name, offset, hp)
	local p = Instance.new("Part")
	p.Name = name
	p.Anchored = true
	p.CanCollide = false
	p.Size = Vector3.new(2, 2, 2)
	p.Position = root.Position + offset
	p:SetAttribute("Health", hp)
	CollectionService:AddTag(p, Action.TAG)
	p.Parent = scratch
	task.wait()
	return p, p:GetAttribute("__NAME__Id")
end
local clock = 1000
Action.now = function() return clock end
local function score() return player:GetAttribute("__NAME__Score") or 0 end
local function case(name, want, id, step)
	clock += step or (Action.COOLDOWN + 0.01)
	local s0 = score()
	local ok, why = Action.handle(player, id)
	local paid = score() - s0
	table.insert(cases, { name = name, pass = ok == want and paid == (if want then 1 else 0), got = { accepted = ok, why = why, paid = paid } })
end
local near, nearId = target("SelfTestNear", Vector3.new(4, 0, 0), 100)
local near2, near2Id = target("SelfTestNear2", Vector3.new(-4, 0, 0), 100)
local far, farId = target("SelfTestFar", Vector3.new(200, 0, 0), 100)
local dead, deadId = target("SelfTestDead", Vector3.new(0, 0, 4), 0)
case("a target in reach pays", true, nearId)
case("again inside the cooldown pays nothing", false, nearId, 0.01)
case("a second target named inside the cooldown pays nothing", false, near2Id, 0.01)
case("a target out of reach pays nothing", false, farId)
case("a miss spends the cooldown: a good target right after pays nothing", false, nearId, 0.01)
case("a dead target pays nothing", false, deadId)
case("a number as the id pays nothing", false, 12345)
case("NaN as the id pays nothing", false, 0 / 0)
case("a table as the id pays nothing", false, { nearId })
case("an unknown name pays nothing", false, "NoSuchTarget")
case("the part itself as the id pays nothing", false, near)
case("nil pays nothing", false, nil)
case("after the cooldown a target in reach pays again", true, nearId)
-- a wall: the 1-stud cell holding the point 2 studs out towards the near target, which the line always crosses (found
-- live 2026-11-26: a wall of cells whose centre lay strictly between 1.5 and 2.5 studs out was empty when the root
-- stood on a whole stud, so the case passed a swing through it); none towards near2
local wallCell = math.floor(root.Position.X + 2)
Action.GRID = { CELL = 1, solid = function(x, _y, _z) return x == wallCell end }
case("a target behind another block pays nothing", false, nearId)
case("a target with a clear line pays", true, near2Id)
Action.GRID = { CELL = 1, solid = function() error("grid broke") end }
case("a grid that errors pays nothing (unread is refused)", false, near2Id)
Action.GRID = nil
Action.now = os.clock
for _, p in ipairs({ near, near2, far, dead }) do p:Destroy() end
report()
'''

# ---------------------------------------------------------------- template "prompt_once"
PROMPT_MODULE = r'''-- __NAME__Prompt: a prompt (favorite or notifications) shown once per player, at a moment the server chooses.
-- Written by MifBridge's rbx_script_scaffold (kind prompt_once). The server claims the prompt and saves the claim before
-- the client is told; never in a player's first minutes, never while another prompt or menu is up; a claim that did
-- not save shows nothing and can be tried again. The client calls no API when game.PlaceId is 0 (an unpublished place).

local Prompt = {}
Prompt.KIND = "__KIND__"            -- "favorite" or "notifications"
Prompt.STORE_NAME = "__NAME__PromptClaims"
Prompt.MIN_SECONDS = 180            -- never in the first 3 minutes of a session
Prompt.BUSY_ATTRIBUTES = { "PromptOpen", "PurchasePromptOpen", "MenuOpen", "RevealOpen" } -- set true on the player while busy
Prompt.now = os.clock
Prompt.fire = nil                   -- set by __NAME__PromptServer: function(player) tells the client to prompt

__SEAM__
local ds = nil
local function store()
	ds = ds or dataStores():GetDataStore(Prompt.STORE_NAME)
	return ds
end
local joined = {}  -- player -> Prompt.now() at join
local claimed = {} -- userId -> true once a claim saved (this server's cache)

function Prompt.track(player)
	joined[player] = Prompt.now()
end
function Prompt.forget(player)
	joined[player] = nil
	claimed[player.UserId] = nil
end

-- Offer the prompt now. Answers shown (bool) and why.
function Prompt.offer(player)
	if joined[player] == nil then return false, "not tracked" end
	if Prompt.now() - joined[player] < Prompt.MIN_SECONDS then return false, "too early" end
	for _, a in ipairs(Prompt.BUSY_ATTRIBUTES) do
		if player:GetAttribute(a) then return false, "busy: " .. a end
	end
	if claimed[player.UserId] then return false, "already claimed" end
	local was = nil
	local ok, err = pcall(function()
		store():UpdateAsync("Player_" .. tostring(player.UserId), function(old)
			was = old
			if old then return nil end -- claimed before (another session or server): leave it
			return { At = os.time() }
		end)
	end)
	if not ok then return false, "the claim did not save: " .. tostring(err) end
	claimed[player.UserId] = true
	if was then return false, "already claimed" end
	if Prompt.fire then Prompt.fire(player) end
	return true, "shown"
end

return Prompt
'''

PROMPT_SERVER = r'''-- __NAME__PromptServer: tracks players and tells a client to prompt. Call Prompt.offer(player) at your reward moment
-- (a first rare find, a level-up). Written by MifBridge's rbx_script_scaffold.
local RS = game:GetService("ReplicatedStorage")
local Players = game:GetService("Players")
local Prompt = require(game:GetService("ServerScriptService"):WaitForChild("__NAME__Prompt"))
local remote = RS:WaitForChild("__NAME__PromptRemote")
Prompt.fire = function(player)
	remote:FireClient(player, Prompt.KIND)
end
Players.PlayerAdded:Connect(Prompt.track)
for _, p in ipairs(Players:GetPlayers()) do Prompt.track(p) end
Players.PlayerRemoving:Connect(Prompt.forget)
'''

PROMPT_CLIENT = r'''-- __NAME__PromptClient: shows the prompt the server claimed. No API call when game.PlaceId is 0 (Studio, unpublished).
-- Written by MifBridge's rbx_script_scaffold.
local RS = game:GetService("ReplicatedStorage")
local remote = RS:WaitForChild("__NAME__PromptRemote")
remote.OnClientEvent:Connect(function(kind)
	if game.PlaceId == 0 then
		print("__NAME__Prompt: would prompt " .. tostring(kind) .. " (PlaceId is 0, so no API call)")
		return
	end
	if kind == "favorite" then
		local AES = game:GetService("AvatarEditorService")
		-- GetFavoriteAsync errors unless the player granted inventory access (ledger RB1193): skip only on a clear true
		local ok, isFav = pcall(function() return AES:GetFavoriteAsync(game.PlaceId, Enum.AvatarItemType.Asset) end)
		if ok and isFav == true then return end
		pcall(function() AES:PromptSetFavorite(game.PlaceId, Enum.AvatarItemType.Asset, true) end)
	elseif kind == "notifications" then
		local ENS = game:GetService("ExperienceNotificationService")
		local ok, can = pcall(function() return ENS:CanPromptOptInAsync() end)
		if ok and can then pcall(function() ENS:PromptOptIn() end) end
	end
end)
'''

PROMPT_SELFTEST = r'''-- __NAME__PromptSelfTest: the prompt template's self-test. Runs only in Studio, only while MifBridge's play probe
-- has put Workspace.MifScratch.SelfTest___NAME__ and the DataStore stand-in there; it needs one player in the test.
local RunService = game:GetService("RunService")
local HttpService = game:GetService("HttpService")
local Players = game:GetService("Players")
if not RunService:IsStudio() then return end
local scratch = workspace:WaitForChild("MifScratch", 5)
if scratch == nil or scratch:WaitForChild("SelfTest___NAME__", 20) == nil then return end
local S = require(scratch:WaitForChild("DataStoreStandIn"))
local Prompt = require(game:GetService("ServerScriptService"):WaitForChild("__NAME__Prompt"))
local cases = {}
local t0 = os.clock()
while #Players:GetPlayers() < 1 and os.clock() - t0 < 20 do task.wait(0.2) end
local player = Players:GetPlayers()[1]
local clock = 1000
Prompt.now = function() return clock end
local shown = 0
local realFire = Prompt.fire
Prompt.fire = function() shown += 1 end
local function case(name, wantShown, wantShownCount)
	local ok, why = Prompt.offer(player)
	table.insert(cases, { name = name, pass = ok == wantShown and shown == wantShownCount, got = { shown = ok, why = why, fired = shown } })
end
if player then
	Prompt.forget(player)
	Prompt.track(player)
	clock += 10
	case("not in the first minutes", false, 0)
	clock += Prompt.MIN_SECONDS
	player:SetAttribute(Prompt.BUSY_ATTRIBUTES[1], true)
	case("not over another prompt or menu", false, 0)
	player:SetAttribute(Prompt.BUSY_ATTRIBUTES[1], nil)
	S.schedule({ op = "UpdateAsync", store = Prompt.STORE_NAME, key = "Player_" .. tostring(player.UserId), nth = 1, count = 1,
		["do"] = "fail", after = 0 })
	case("a claim that did not save shows nothing", false, 0)
	case("then it is shown once, after the claim saved", true, 1)
	local rec = S.peek(Prompt.STORE_NAME, "Player_" .. tostring(player.UserId))
	table.insert(cases, { name = "the claim is in the store before the client is told", pass = rec ~= nil and shown == 1, got = rec })
	case("never twice on this server", false, 1)
	Prompt.forget(player)
	Prompt.track(player)
	clock += Prompt.MIN_SECONDS + 1
	case("never twice after a rejoin (read from the store)", false, 1)
else
	table.insert(cases, { name = "a player joined", pass = false, got = "no player in 20 s" })
end
Prompt.now = os.clock
Prompt.fire = realFire
local passed = 0
for _, c in ipairs(cases) do if c.pass then passed += 1 end end
local j = HttpService:JSONEncode({ template = "prompt_once", name = "__NAME__", passed = passed, total = #cases, cases = cases })
--@emit MIFSELFTEST __NAME__.prompt_once
'''


# ---------------------------------------------------------------- template "economy"
ECONOMY_CLOCK = r'''-- __NAME__Clock: the one reader of game time. Written by MifBridge's rbx_script_scaffold (kind economy).
-- Every script that pays, counts down or resets by time asks this module, never os.time, os.clock, tick or DateTime
-- itself, so a test can move game time for all of them at once: offset shifts it, freeze holds it still.
local Clock = {}
Clock.offset = 0                     -- seconds added to real time; a test moves it
Clock.frozen = nil                   -- a fixed real time while a test holds the clock still
Clock.RESET_HOUR_UTC = 0             -- the daily reset, in hours after midnight UTC
Clock.WEEK_STARTS = 1                -- 1 Monday ... 7 Sunday: the day the weekly reset falls on

function Clock.real()
	return DateTime.now().UnixTimestampMillis / 1000
end

function Clock.now()
	return (Clock.frozen or Clock.real()) + Clock.offset
end

function Clock.advance(seconds)
	Clock.offset += seconds
end

-- Hold game time at t (unix seconds) until release; a soak steps it with advance.
function Clock.freeze(t)
	Clock.frozen = t
	Clock.offset = 0
end

function Clock.release()
	Clock.frozen = nil
	Clock.offset = 0
end

-- The day number of t: it goes up by one at each daily reset.
function Clock.day(t)
	return math.floor((t - Clock.RESET_HOUR_UTC * 3600) / 86400)
end

-- The week number of t: it goes up by one at the daily reset of WEEK_STARTS. Day 0 (1970-01-01) was a Thursday (4).
function Clock.week(t)
	return math.floor((Clock.day(t) + 4 - Clock.WEEK_STARTS) / 7)
end

return Clock
'''

ECONOMY_MODULE = r'''-- __NAME__Economy: income per second, timed boosts, offline pay with a cap, and the daily and weekly resets.
-- Written by MifBridge's rbx_script_scaffold (kind economy). A player's economy is one plain table (state): keep it in
-- the player's save profile (the save template's data) so it survives a rejoin. Time comes only from __NAME__Clock.
--
-- THE ORDER IN A TICK: pay the income first, with the boosts that were running, THEN count the boosts down. The other
-- order pays every timed boost one tick short (found on a game by this template's own soak: a 30 minute boost doubled
-- 29 min 59 s). A boost's extra is paid for min(time left, the tick's length), so it pays exactly its length however
-- the ticks fall.
local Clock = require(script.Parent:WaitForChild("__NAME__Clock"))

local Economy = {}
Economy.RATE = 1                     -- income per second with no boost
Economy.OFFLINE_RATE = 0.5           -- the share of RATE paid while away
Economy.OFFLINE_CAP = 8 * 3600       -- the most seconds of offline pay
Economy.TICK = 1                     -- real seconds between ticks (__NAME__EconomyServer)
Economy.onDay = function(state, day, skipped) end   -- the daily reset: once per crossing; skipped >= 1 days passed
Economy.onWeek = function(state, week, skipped) end -- the weekly reset, likewise

function Economy.new(t)
	t = t or Clock.now()
	return { coins = 0, rate = Economy.RATE, boosts = {}, last = t, seen = t, day = Clock.day(t), week = Clock.week(t),
		days = 0, weeks = 0 }
end

-- Income for dt seconds: the rate, plus each running boost's extra over the seconds it still has in this tick.
function Economy.payFor(state, dt)
	local got = state.rate * dt
	for _, b in ipairs(state.boosts) do
		local extra = state.rate * (b.mult - 1) * math.min(math.max(b.left, 0), dt)
		b.paid += extra
		got += extra
	end
	state.coins += got
	return got
end

function Economy.countDown(state, dt)
	for i = #state.boosts, 1, -1 do
		local b = state.boosts[i]
		b.left -= dt
		if b.left <= 0 then table.remove(state.boosts, i) end
	end
end

local function rollovers(state, t)
	local d, w = Clock.day(t), Clock.week(t)
	if d > state.day then
		local skipped = d - state.day
		state.day = d
		state.days += 1
		Economy.onDay(state, d, skipped)
	end
	if w > state.week then
		local skipped = w - state.week
		state.week = w
		state.weeks += 1
		Economy.onWeek(state, w, skipped)
	end
end

-- Pay up to now. A clock that has not moved, or moved back, pays nothing and counts nothing.
function Economy.tick(state)
	local t = Clock.now()
	local dt = t - state.last
	if dt <= 0 then return 0 end
	state.last = t
	state.seen = t
	local got = Economy.payFor(state, dt) -- income first ...
	Economy.countDown(state, dt)          -- ... then the boosts count down
	rollovers(state, t)
	return got
end

-- A timed boost: income times mult for seconds. Settles the income up to now first, so it starts now.
function Economy.boost(state, mult, seconds, name)
	Economy.tick(state)
	local b = { mult = mult, left = seconds, seconds = seconds, name = name, paid = 0 }
	table.insert(state.boosts, b)
	return b
end

-- A join: offline pay for the time since the state was last seen (never negative, at most OFFLINE_CAP), then the
-- resets that fell while away, once each. Boosts do not count down while away.
function Economy.join(state)
	local t = Clock.now()
	local away = math.clamp(t - (state.seen or t), 0, Economy.OFFLINE_CAP)
	local got = state.rate * Economy.OFFLINE_RATE * away
	state.coins += got
	state.last = math.max(state.last, t)
	state.seen = state.last
	rollovers(state, t)
	return got, away
end

-- A leave: pay up to now; seen is now, so the next join's offline pay starts here.
function Economy.leave(state)
	return Economy.tick(state)
end

return Economy
'''

ECONOMY_SERVER = r'''-- __NAME__EconomyServer: ticks every player's economy. Written by MifBridge's rbx_script_scaffold (kind economy).
-- load and save are where a game's save layer plugs in (the save template's profile data): load answers a stored state
-- table or nil, save stores one. With neither, every join starts a fresh economy.
local Players = game:GetService("Players")
local Economy = require(game:GetService("ServerScriptService"):WaitForChild("__NAME__Economy"))

Economy.load = Economy.load or function(player) return nil end
Economy.save = Economy.save or function(player, state) end
Economy.states = Economy.states or {}

local function show(player, state)
	player:SetAttribute("__NAME__Coins", math.floor(state.coins))
end

local function joined(player)
	local state = Economy.load(player) or Economy.new()
	Economy.states[player] = state
	Economy.join(state)
	show(player, state)
end

Players.PlayerAdded:Connect(joined)
for _, p in ipairs(Players:GetPlayers()) do task.spawn(joined, p) end
Players.PlayerRemoving:Connect(function(player)
	local state = Economy.states[player]
	if state then
		Economy.leave(state)
		Economy.save(player, state)
		Economy.states[player] = nil
	end
end)

while true do
	task.wait(Economy.TICK)
	for player, state in pairs(Economy.states) do
		Economy.tick(state)
		show(player, state)
	end
end
'''

ECONOMY_SELFTEST = r'''-- __NAME__EconomySelfTest: the economy template's time-warp soak. Runs only in Studio, only while MifBridge's play
-- probe has put Workspace.MifScratch.SelfTest___NAME__ there. It holds __NAME__Clock still at a Sunday 20:00 UTC,
-- steps it 10 s at a time for 8 hours on a fake player (a plain state, no Player) and compares the earnings with the
-- closed form, counts the day and week resets (each exactly once across the Sunday-to-Monday midnight), then checks
-- offline pay and its cap. It never yields while the clock is moved, and gives the clock back before it reports.
local RunService = game:GetService("RunService")
local HttpService = game:GetService("HttpService")
if not RunService:IsStudio() then return end
local scratch = workspace:WaitForChild("MifScratch", 5)
if scratch == nil or scratch:WaitForChild("SelfTest___NAME__", 20) == nil then return end
local SSS = game:GetService("ServerScriptService")
local Clock = require(SSS:WaitForChild("__NAME__Clock"))
local Economy = require(SSS:WaitForChild("__NAME__Economy"))
local cases = {}
local function case(name, pass, got)
	table.insert(cases, { name = name, pass = pass == true, got = got })
end
local function near(a, b)
	return math.abs(a - b) <= 1e-6 * math.max(1, math.abs(b))
end
local function report()
	local passed = 0
	for _, c in ipairs(cases) do if c.pass then passed += 1 end end
	local j = HttpService:JSONEncode({ template = "economy", name = "__NAME__", passed = passed, total = #cases, cases = cases })
	--@emit MIFSELFTEST __NAME__.economy
end

local keep = { rate = Economy.RATE, offRate = Economy.OFFLINE_RATE, cap = Economy.OFFLINE_CAP, onDay = Economy.onDay,
	onWeek = Economy.onWeek, reset = Clock.RESET_HOUR_UTC, starts = Clock.WEEK_STARTS }
local ok, err = pcall(function()
	Clock.RESET_HOUR_UTC, Clock.WEEK_STARTS = 0, 1
	Economy.RATE, Economy.OFFLINE_RATE, Economy.OFFLINE_CAP = 3, 0.5, 8 * 3600
	local resets = {}
	Economy.onDay = function(_s, day, skipped) table.insert(resets, { kind = "day", n = day, skipped = skipped }) end
	Economy.onWeek = function(_s, week, skipped) table.insert(resets, { kind = "week", n = week, skipped = skipped }) end
	local t0 = DateTime.fromUniversalTime(2026, 11, 22, 20, 0, 0).UnixTimestamp -- a Sunday, 20:00 UTC
	case("the start is a Sunday evening: Monday's midnight is the next day and the next week",
		Clock.day(t0 + 4 * 3600) == Clock.day(t0) + 1 and Clock.week(t0 + 4 * 3600) == Clock.week(t0) + 1
			and Clock.week(t0 - 6 * 86400) == Clock.week(t0),
		{ day = Clock.day(t0), week = Clock.week(t0) })
	Clock.freeze(t0)
	local s = Economy.new()
	local R, STEP, HOURS = Economy.RATE, 10, 8
	-- boosts bought on a tick, between ticks, and one that outlasts a tick by 5 s; mult, seconds, at (seconds from t0)
	local plan = { { 2, 1800, 0 }, { 3, 900, 3600 }, { 2, 1795, 3 * 3600 + 5 }, { 1.5, 125, 5 * 3600 + 7 } }
	local made, k = {}, 1
	local steps = HOURS * 3600 / STEP
	for _ = 1, steps do
		local nextAt = Clock.offset + STEP
		while plan[k] and plan[k][3] < nextAt do
			local p = plan[k]
			Clock.offset = p[3] -- buy at that moment: boost settles the income up to it
			made[k] = Economy.boost(s, p[1], p[2], "b" .. k)
			k += 1
		end
		Clock.offset = nextAt
		Economy.tick(s)
	end
	local T = HOURS * 3600
	local want = R * T
	for _, p in ipairs(plan) do want += R * (p[1] - 1) * p[2] end
	case("8 h in 10 s steps pays the closed form", near(s.coins, want), { coins = s.coins, want = want, steps = steps })
	local short = {}
	for i, b in ipairs(made) do
		local w = R * (b.mult - 1) * b.seconds
		if not near(b.paid, w) then table.insert(short, { boost = i, paid = b.paid, want = w }) end
	end
	case("every boost paid its whole length, none a tick short", #short == 0 and #made == #plan, short)
	case("every boost ran out and was removed", #s.boosts == 0, #s.boosts)
	local days, weeks = 0, 0
	for _, r in ipairs(resets) do
		if r.kind == "day" then days += 1 else weeks += 1 end
	end
	case("the daily reset fired exactly once, at Monday's midnight", days == 1 and s.days == 1, resets)
	case("the weekly reset fired exactly once, at Monday's midnight", weeks == 1 and s.weeks == 1, resets)
	local c0 = s.coins
	local again = Economy.tick(s)
	case("a tick with the clock unmoved pays nothing and resets nothing", again == 0 and s.coins == c0 and #resets == 2,
		{ paid = again, resets = #resets })
	Clock.advance(-600)
	local back = Economy.tick(s)
	Clock.advance(600)
	local same = Economy.tick(s)
	case("a clock moved back pays nothing, and nothing twice when it comes forward again",
		back == 0 and same == 0 and s.coins == c0, { back = back, forward = same })
	-- the soak catches the bug it exists for: the same run with the countdown first pays each boost one tick short
	local bad = Economy.new(t0)
	local badMade = {}
	for i = 1, steps do
		if i == 1 then
			badMade[1] = { mult = 2, left = 1800, seconds = 1800, paid = 0 }
			table.insert(bad.boosts, badMade[1])
		end
		Economy.countDown(bad, STEP)
		Economy.payFor(bad, STEP)
	end
	case("the soak tells the wrong order: counting down first pays a 30 min boost one tick short",
		near(badMade[1].paid, R * 1 * (1800 - STEP)), { paid = badMade[1].paid, whole = R * 1800 })
	-- offline pay: 2 h away pays RATE x OFFLINE_RATE x 7200; 3 days away stops at the cap and resets once each
	Economy.leave(s)
	local before = #resets
	Clock.advance(2 * 3600)
	local got, away = Economy.join(s)
	case("2 h away pays rate x offline rate x 7200", near(got, R * 0.5 * 7200) and away == 7200, { paid = got, away = away })
	Economy.leave(s)
	Clock.advance(3 * 86400)
	got, away = Economy.join(s)
	local fired = {}
	for i = before + 1, #resets do table.insert(fired, resets[i]) end
	case("3 days away pays only the cap", near(got, R * 0.5 * Economy.OFFLINE_CAP) and away == Economy.OFFLINE_CAP,
		{ paid = got, away = away })
	case("3 days away fires the daily reset once, told it skipped 3 days", #fired >= 1 and fired[1].kind == "day"
		and fired[1].skipped == 3 and s.days == 2, fired)
	Economy.leave(s)
	local c1 = s.coins
	Clock.advance(-3600)
	got, away = Economy.join(s)
	case("a join with the clock moved back pays nothing", got == 0 and away == 0 and s.coins == c1, { paid = got })
end)
Clock.release()
Economy.RATE, Economy.OFFLINE_RATE, Economy.OFFLINE_CAP = keep.rate, keep.offRate, keep.cap
Economy.onDay, Economy.onWeek = keep.onDay, keep.onWeek
Clock.RESET_HOUR_UTC, Clock.WEEK_STARTS = keep.reset, keep.starts
if not ok then case("the soak ran to the end", false, tostring(err)) end
report()
'''


# ---------------------------------------------------------------- template "scenery"
SCENERY_TEXTURES = r'''-- __NAME__Textures: the pixel data every scenery texture is drawn from, shipped in the place (no upload, no asset id).
-- Written by MifBridge's rbx_script_scaffold (kind scenery). A build script replaces this table with its own drawn
-- textures; each entry is { w = width, h = height, rgba = one 8-digit hex RRGGBBAA per pixel, row by row from the top }.
-- A texture with transparent pixels (bars, a grate) goes on a part whose Transparency is 1, so only the texture shows.
local Textures = {}
Textures.stone = { w = 4, h = 4, rgba =
	"7A7A7AFF8A8A8AFF7A7A7AFF6E6E6EFF" ..
	"8A8A8AFF7A7A7AFF6E6E6EFF7A7A7AFF" ..
	"6E6E6EFF7A7A7AFF8A8A8AFF7A7A7AFF" ..
	"7A7A7AFF6E6E6EFF7A7A7AFF8A8A8AFF" }
Textures.bars = { w = 4, h = 4, rgba =
	"30303AFF000000000000000030303AFF" ..
	"30303AFF000000000000000030303AFF" ..
	"30303AFF30303AFF30303AFF30303AFF" ..
	"30303AFF000000000000000030303AFF" }
return Textures
'''

SCENERY_MODULE = r'''-- __NAME__Scenery: the tag convention and the pixel helpers both halves share. Written by MifBridge's
-- rbx_script_scaffold (kind scenery). A scenery part carries CollectionService tags:
--   __NAME__Tex                the part is scenery
--   __NAME__Tex=<id>           the texture on every face but the bottom (an id in __NAME__Textures)
--   __NAME__Top=<id>           optional: another texture for the top face
--   __NAME__Tile=<studs>       optional: the tile size, StudsPerTileU and V (default DEFAULT_TILE)
local CollectionService = game:GetService("CollectionService")
local Scenery = {}
Scenery.TAG = "__NAME__Tex"
Scenery.TEX = "__NAME__Tex="
Scenery.TOP = "__NAME__Top="
Scenery.TILE = "__NAME__Tile="
Scenery.DEFAULT_TILE = 4

-- texture id, top id (or nil), tile studs, for one tagged part
function Scenery.parse(part)
	local tex, top, tile = nil, nil, Scenery.DEFAULT_TILE
	for _, t in ipairs(CollectionService:GetTags(part)) do
		if string.sub(t, 1, #Scenery.TEX) == Scenery.TEX then
			tex = string.sub(t, #Scenery.TEX + 1)
		elseif string.sub(t, 1, #Scenery.TOP) == Scenery.TOP then
			top = string.sub(t, #Scenery.TOP + 1)
		elseif string.sub(t, 1, #Scenery.TILE) == Scenery.TILE then
			tile = tonumber(string.sub(t, #Scenery.TILE + 1)) or tile
		end
	end
	return tex, top, tile
end

-- the entry as a buffer of w * h * 4 bytes (RGBA), or nil and why
function Scenery.decode(entry)
	if type(entry) ~= "table" or type(entry.w) ~= "number" or type(entry.h) ~= "number" or type(entry.rgba) ~= "string" then
		return nil, "not { w, h, rgba }"
	end
	local n = entry.w * entry.h
	if #entry.rgba ~= n * 8 then
		return nil, string.format("rgba holds %d hex digits; %d x %d pixels need %d", #entry.rgba, entry.w, entry.h, n * 8)
	end
	local b = buffer.create(n * 4)
	for i = 0, n * 4 - 1 do
		local v = tonumber(string.sub(entry.rgba, i * 2 + 1, i * 2 + 2), 16)
		if v == nil then
			return nil, "rgba is not hex at pixel " .. tostring(i // 4)
		end
		buffer.writeu8(b, i, v)
	end
	return b
end

-- the average colour of the opaque pixels (what the server paints a part), and the share of pixels that are opaque
function Scenery.average(entry)
	local b, why = Scenery.decode(entry)
	if b == nil then
		return nil, why
	end
	local r, g, bl, k = 0, 0, 0, 0
	local n = entry.w * entry.h
	for i = 0, n - 1 do
		if buffer.readu8(b, i * 4 + 3) >= 128 then
			r += buffer.readu8(b, i * 4)
			g += buffer.readu8(b, i * 4 + 1)
			bl += buffer.readu8(b, i * 4 + 2)
			k += 1
		end
	end
	if k == 0 then
		return Color3.new(0, 0, 0), 0
	end
	return Color3.fromRGB(math.floor(r / k + 0.5), math.floor(g / k + 0.5), math.floor(bl / k + 0.5)), k / n
end

-- every tagged part under root (default the whole place) whose texture id has no pixel data, and every texture that
-- does not decode: { parts = n, missing = { { path, id } }, bad = { { id, why } } }
function Scenery.audit(textures, root)
	local out = { parts = 0, missing = {}, bad = {} }
	for id, entry in pairs(textures) do
		local b, why = Scenery.decode(entry)
		if b == nil then
			table.insert(out.bad, { id = id, why = why })
		end
	end
	for _, part in ipairs(CollectionService:GetTagged(Scenery.TAG)) do
		if root == nil or part:IsDescendantOf(root) then
			out.parts += 1
			local tex, top = Scenery.parse(part)
			if tex == nil then
				table.insert(out.missing, { path = part:GetFullName(), id = "(no " .. Scenery.TEX .. " tag)" })
			elseif textures[tex] == nil then
				table.insert(out.missing, { path = part:GetFullName(), id = tex })
			end
			if top ~= nil and textures[top] == nil then
				table.insert(out.missing, { path = part:GetFullName(), id = top })
			end
		end
	end
	return out
end

return Scenery
'''

SCENERY_SERVER = r'''-- __NAME__SceneryServer: paints every scenery part the average colour of its texture, so edit mode and a client that
-- cannot make EditableImages still read right; says in the Output which tagged ids have no pixel data; and keeps each
-- client's last tiling report on __NAME__SceneryReport's attributes (Images, Textures, Missing, Failed, FirstError).
-- Written by MifBridge's rbx_script_scaffold (kind scenery).
local CollectionService = game:GetService("CollectionService")
local RS = game:GetService("ReplicatedStorage")
local Scenery = require(RS:WaitForChild("__NAME__Scenery"))
local Textures = require(RS:WaitForChild("__NAME__Textures"))
local report = RS:WaitForChild("__NAME__SceneryReport")
local colours = {}
local function paint(part)
	if not part:IsA("BasePart") then
		return
	end
	local tex = Scenery.parse(part)
	if tex == nil or Textures[tex] == nil then
		return
	end
	if colours[tex] == nil then
		colours[tex] = Scenery.average(Textures[tex]) or false
	end
	if colours[tex] then
		part.Color = colours[tex]
	end
end
for _, p in ipairs(CollectionService:GetTagged(Scenery.TAG)) do
	paint(p)
end
CollectionService:GetInstanceAddedSignal(Scenery.TAG):Connect(paint)
local a = Scenery.audit(Textures)
for _, m in ipairs(a.missing) do
	warn("__NAME__Scenery: " .. m.path .. " names texture " .. m.id .. ", which has no pixel data")
end
for _, b in ipairs(a.bad) do
	warn("__NAME__Scenery: texture " .. tostring(b.id) .. " does not decode: " .. b.why)
end
report.OnServerEvent:Connect(function(player, r)
	if type(r) ~= "table" then
		return
	end
	report:SetAttribute("Images", tonumber(r.images) or 0)
	report:SetAttribute("Textures", tonumber(r.textures) or 0)
	report:SetAttribute("Missing", tonumber(r.missing) or 0)
	report:SetAttribute("Failed", tonumber(r.failed) or 0)
	report:SetAttribute("FirstError", if type(r.firstError) == "string" then string.sub(r.firstError, 1, 200) else "")
	report:SetAttribute("From", player.UserId)
end)
'''

SCENERY_CLIENT = r'''-- __NAME__SceneryClient: tiles every scenery part on this client: one EditableImage per texture id, a Texture on every
-- face but the bottom (StudsPerTileU/V the tile size, ResampleMode Pixelated). Written by MifBridge's rbx_script_scaffold
-- (kind scenery). A client that cannot make an EditableImage keeps the server's average colour. It tells the server
-- what it made over __NAME__SceneryReport after its first pass and after each part added later.
local CollectionService = game:GetService("CollectionService")
local AssetService = game:GetService("AssetService")
local RS = game:GetService("ReplicatedStorage")
local Scenery = require(RS:WaitForChild("__NAME__Scenery"))
local Textures = require(RS:WaitForChild("__NAME__Textures"))
local report = RS:WaitForChild("__NAME__SceneryReport")
local images = {}
local counts = { images = 0, textures = 0, missing = 0, failed = 0, firstError = nil }
local function fail(why)
	counts.failed += 1
	counts.firstError = counts.firstError or why
end
local function image(id)
	if images[id] ~= nil then
		return images[id] or nil
	end
	local entry = Textures[id]
	if entry == nil then
		images[id] = false
		counts.missing += 1
		return nil
	end
	local b, why = Scenery.decode(entry)
	if b == nil then
		images[id] = false
		fail("texture " .. tostring(id) .. ": " .. tostring(why))
		return nil
	end
	local ok, img = pcall(function()
		local e = AssetService:CreateEditableImage({ Size = Vector2.new(entry.w, entry.h) })
		e:WritePixelsBuffer(Vector2.zero, Vector2.new(entry.w, entry.h), b)
		return e
	end)
	if not ok or img == nil then
		images[id] = false
		fail("CreateEditableImage: " .. tostring(img))
		return nil
	end
	images[id] = img
	counts.images += 1
	return img
end
local FACES = { Enum.NormalId.Top, Enum.NormalId.Front, Enum.NormalId.Back, Enum.NormalId.Left, Enum.NormalId.Right }
local function tile(part)
	if not part:IsA("BasePart") or part:FindFirstChild("__NAME__Tile") then
		return
	end
	local tex, top, size = Scenery.parse(part)
	if tex == nil then
		return
	end
	for _, face in ipairs(FACES) do
		local img = image(if face == Enum.NormalId.Top and top ~= nil then top else tex)
		if img ~= nil then
			local ok, err = pcall(function()
				local t = Instance.new("Texture")
				t.Name = "__NAME__Tile"
				t.Face = face
				t.StudsPerTileU, t.StudsPerTileV = size, size
				t.ResampleMode = Enum.ResamplerMode.Pixelated
				t.ColorMapContent = Content.fromObject(img)
				t.Parent = part
			end)
			if ok then
				counts.textures += 1
			else
				fail("Texture: " .. tostring(err))
			end
		end
	end
end
local function send()
	report:FireServer(counts)
end
for _, p in ipairs(CollectionService:GetTagged(Scenery.TAG)) do
	tile(p)
end
CollectionService:GetInstanceAddedSignal(Scenery.TAG):Connect(function(p)
	tile(p)
	send()
end)
send()
'''

SCENERY_SELFTEST = r'''-- __NAME__ScenerySelfTest: the scenery kit's self-test. Runs only in Studio, only while MifBridge's play probe has put
-- Workspace.MifScratch.SelfTest___NAME__ there; it needs one player, whose client tiles the parts it plants.
local RunService = game:GetService("RunService")
local HttpService = game:GetService("HttpService")
local CollectionService = game:GetService("CollectionService")
local Players = game:GetService("Players")
if not RunService:IsStudio() then
	return
end
local scratch = workspace:WaitForChild("MifScratch", 5)
if scratch == nil or scratch:WaitForChild("SelfTest___NAME__", 20) == nil then
	return
end
local RS = game:GetService("ReplicatedStorage")
local Scenery = require(RS:WaitForChild("__NAME__Scenery"))
local Textures = require(RS:WaitForChild("__NAME__Textures"))
local report = RS:WaitForChild("__NAME__SceneryReport")
local cases = {}
local function case(name, pass, got)
	table.insert(cases, { name = name, pass = pass == true, got = got })
end
local t0 = os.clock()
while #Players:GetPlayers() < 1 and os.clock() - t0 < 20 do
	task.wait(0.2)
end
local bad = {}
for id, entry in pairs(Textures) do
	local b, why = Scenery.decode(entry)
	if b == nil or buffer.len(b) ~= entry.w * entry.h * 4 then
		table.insert(bad, { id = id, why = why })
	end
end
case("every texture in __NAME__Textures decodes to w x h RGBA pixels", #bad == 0, bad)
local avg, share = Scenery.average({ w = 2, h = 1, rgba = "FF0000FF0000FF40" })
case("the average is of opaque pixels only: opaque red beside a faint blue gives red, half the pixels opaque",
	avg ~= nil and avg:ToHex() == "ff0000" and share == 0.5, { avg = avg and avg:ToHex(), share = share })
local wrong = Scenery.decode({ w = 2, h = 2, rgba = "FF" })
case("a texture whose hex is the wrong length is refused", wrong == nil, wrong)
local firstId = nil
for id in pairs(Textures) do
	if firstId == nil or tostring(id) < tostring(firstId) then
		firstId = id
	end
end
local before = report:GetAttribute("Textures") or 0
local own = Instance.new("Folder")  -- the self-test's own parts, so a build's scenery elsewhere is never counted here
own.Name = "__NAME__SceneryTest"
own.Parent = scratch
local good = Instance.new("Part")
good.Name = "__NAME__SceneryGood"
good.Anchored, good.Size, good.Position = true, Vector3.new(8, 4, 8), Vector3.new(0, 200, 0)
CollectionService:AddTag(good, Scenery.TAG)
CollectionService:AddTag(good, Scenery.TEX .. tostring(firstId))
CollectionService:AddTag(good, Scenery.TILE .. "2")
local ghost = Instance.new("Part")
ghost.Name = "__NAME__SceneryGhost"
ghost.Anchored, ghost.Position = true, Vector3.new(20, 200, 0)
CollectionService:AddTag(ghost, Scenery.TAG)
CollectionService:AddTag(ghost, Scenery.TEX .. "noSuchTexture")
good.Parent, ghost.Parent = own, own
local tex, _, tile = Scenery.parse(good)
case("the tags parse: the texture id and tile size 2", tex == tostring(firstId) and tile == 2, { tex = tex, tile = tile })
local a = Scenery.audit(Textures, own)
case("the audit fails a tagged id with no pixel data and passes the real one",
	#a.missing == 1 and a.missing[1].id == "noSuchTexture" and a.parts == 2, a)
local want = Scenery.average(Textures[firstId])
local t1 = os.clock()
while good.Color:ToHex() ~= want:ToHex() and os.clock() - t1 < 5 do
	task.wait(0.1)
end
case("the server paints a tagged part its texture's average colour", good.Color:ToHex() == want:ToHex(),
	{ got = good.Color:ToHex(), want = want:ToHex() })
local t2 = os.clock()
while (report:GetAttribute("Textures") or 0) < before + 5 and os.clock() - t2 < 20 do
	task.wait(0.2)
end
case("the client tiled the planted part: an EditableImage and a Texture on each of its five faces, nothing failed",
	(report:GetAttribute("Textures") or 0) >= before + 5 and (report:GetAttribute("Images") or 0) >= 1
		and (report:GetAttribute("Failed") or 0) == 0,
	{ images = report:GetAttribute("Images"), textures = report:GetAttribute("Textures"),
		missing = report:GetAttribute("Missing"), failed = report:GetAttribute("Failed"),
		firstError = report:GetAttribute("FirstError") })
own:Destroy()
local passed = 0
for _, c in ipairs(cases) do
	if c.pass then
		passed += 1
	end
end
local j = HttpService:JSONEncode({ template = "scenery", name = "__NAME__", passed = passed, total = #cases, cases = cases })
--@emit MIFSELFTEST __NAME__.scenery
'''



# ---------------------------------------------------------------- template "warps"
WARP_MODULE = r'''-- __NAME__Warps: warps between separate islands, checked on the server. Written by MifBridge's rbx_script_scaffold
-- (kind warps). Each area is its own island under Workspace.<ROOT_NAME>: a Model or Folder named by the area's id,
-- holding a part named ArrivalPad whose front (LookVector) faces into the area. A warp - from a portal or the menu -
-- asks the server, which checks the id is a real destination, the area is built and the player's rank allows it, then
-- lands the character on the pad facing the area. The client only colours what it may not use.
local Warps = {}
Warps.ROOT_NAME = "__NAME__Islands"
Warps.AREAS = { Hub = 0, A = 1, B = 2 } -- every real destination and the rank it needs; an area not built yet is refused
Warps.MIN_GAP = 300                    -- studs of sky between two islands, measured from their parts' extents
Warps.COOLDOWN = 1                     -- seconds between one player's warps
Warps.rankOf = function(player)
	return player:GetAttribute("Rank") or 0
end
local last = {}

function Warps.root()
	return workspace:FindFirstChild(Warps.ROOT_NAME)
end

function Warps.pad(id)
	local r = Warps.root()
	local area = r and r:FindFirstChild(id)
	local pad = area and area:FindFirstChild("ArrivalPad", true)
	if pad ~= nil and pad:IsA("BasePart") then
		return pad
	end
	return nil
end

-- may this player warp to id now? true, or false and why: "not a destination", "unbuilt", "rank", "cooldown"
function Warps.check(player, id)
	if type(id) ~= "string" or Warps.AREAS[id] == nil then
		return false, "not a destination"
	end
	if Warps.pad(id) == nil then
		return false, "unbuilt"
	end
	if (tonumber(Warps.rankOf(player)) or 0) < Warps.AREAS[id] then
		return false, "rank"
	end
	if last[player] ~= nil and os.clock() - last[player] < Warps.COOLDOWN then
		return false, "cooldown"
	end
	return true, "ok"
end

-- warp the player's character onto the area's arrival pad, facing the area. true, or false and why.
function Warps.warp(player, id)
	local ok, why = Warps.check(player, id)
	if not ok then
		return false, why
	end
	local char = player.Character
	local root = char and char:FindFirstChild("HumanoidRootPart")
	if root == nil then
		return false, "no character"
	end
	last[player] = os.clock()
	local pad = Warps.pad(id)
	local stand = pad.Position + Vector3.new(0, pad.Size.Y / 2 + 3, 0)
	local look = Vector3.new(pad.CFrame.LookVector.X, 0, pad.CFrame.LookVector.Z)
	if look.Magnitude < 1e-3 then
		look = Vector3.new(0, 0, -1)
	end
	local target = CFrame.lookAt(stand, stand + look.Unit)
	char:PivotTo(target)
	-- The client owns its character's physics: a position it sent before it heard of the warp can arrive after it and
	-- put the character back (measured: once in two play tests, 403 studs back, RB2163). Look again shortly and put it
	-- on the pad once more if it is not near it.
	task.delay(0.5, function()
		if char.Parent ~= nil and root.Parent ~= nil and (root.Position - stand).Magnitude > 8 then
			char:PivotTo(target)
		end
	end)
	return true, "ok"
end

-- for the menu: { [id] = { built, allowed, need } }
function Warps.status(player)
	local out = {}
	local rank = tonumber(Warps.rankOf(player)) or 0
	for id, need in pairs(Warps.AREAS) do
		out[id] = { built = Warps.pad(id) ~= nil, allowed = rank >= need, need = need }
	end
	return out
end

local function extents(inst)
	local lo, hi = nil, nil
	local function add(p)
		lo = if lo then lo:Min(p) else p
		hi = if hi then hi:Max(p) else p
	end
	for _, d in ipairs(inst:GetDescendants()) do
		if d:IsA("BasePart") then
			local h = d.Size / 2
			for _, s in ipairs({ Vector3.new(1, 1, 1), Vector3.new(1, 1, -1), Vector3.new(1, -1, 1), Vector3.new(1, -1, -1),
				Vector3.new(-1, 1, 1), Vector3.new(-1, 1, -1), Vector3.new(-1, -1, 1), Vector3.new(-1, -1, -1) }) do
				add(d.CFrame:PointToWorldSpace(h * s))
			end
		end
	end
	return lo, hi
end

-- every pair of built islands and the sky between their parts' boxes: { pairs = { { a, b, gap } }, tooClose = { ... } }
function Warps.islandGaps()
	local r = Warps.root()
	local boxes, out = {}, { pairs = {}, tooClose = {} }
	for _, area in ipairs(if r then r:GetChildren() else {}) do
		local lo, hi = extents(area)
		if lo ~= nil then
			table.insert(boxes, { id = area.Name, lo = lo, hi = hi })
		end
	end
	table.sort(boxes, function(x, y)
		return x.id < y.id
	end)
	for i = 1, #boxes do
		for j = i + 1, #boxes do
			local a, b = boxes[i], boxes[j]
			local d = Vector3.new(math.max(0, b.lo.X - a.hi.X, a.lo.X - b.hi.X), math.max(0, b.lo.Y - a.hi.Y, a.lo.Y - b.hi.Y),
				math.max(0, b.lo.Z - a.hi.Z, a.lo.Z - b.hi.Z))
			local row = { a = a.id, b = b.id, gap = math.floor(d.Magnitude * 100 + 0.5) / 100 }
			table.insert(out.pairs, row)
			if d.Magnitude < Warps.MIN_GAP then
				table.insert(out.tooClose, row)
			end
		end
	end
	return out
end

function Warps.forget(player)
	last[player] = nil
end

return Warps
'''

WARP_SERVER = r'''-- __NAME__WarpServer: answers the warp menu and wires every portal. A portal is any part tagged __NAME__Portal with an
-- attribute WarpTo naming the area; it gets a ProximityPrompt if it has none. Written by MifBridge's rbx_script_scaffold.
local RS = game:GetService("ReplicatedStorage")
local Players = game:GetService("Players")
local CollectionService = game:GetService("CollectionService")
local Warps = require(game:GetService("ServerScriptService"):WaitForChild("__NAME__Warps"))
local remote = RS:WaitForChild("__NAME__WarpRemote")
remote.OnServerInvoke = function(player, action, id)
	if action == "status" then
		return Warps.status(player)
	elseif action == "warp" then
		return Warps.warp(player, id)
	end
	return false, "unknown action"
end
local function portal(part)
	if not part:IsA("BasePart") then
		return
	end
	local prompt = part:FindFirstChildOfClass("ProximityPrompt")
	if prompt == nil then
		prompt = Instance.new("ProximityPrompt")
		prompt.ActionText = "Warp"
		prompt.ObjectText = tostring(part:GetAttribute("WarpTo") or "")
		prompt.Parent = part
	end
	prompt.Triggered:Connect(function(player)
		Warps.warp(player, part:GetAttribute("WarpTo"))
	end)
end
for _, p in ipairs(CollectionService:GetTagged("__NAME__Portal")) do
	portal(p)
end
CollectionService:GetInstanceAddedSignal("__NAME__Portal"):Connect(portal)
Players.PlayerRemoving:Connect(Warps.forget)
'''

WARP_CLIENT = r'''-- __NAME__WarpClient: a warp menu. It asks the server what each area is (built, allowed, the rank it needs) and only
-- colours what this player may not use; every warp is the server's decision. Written by MifBridge's rbx_script_scaffold.
local RS = game:GetService("ReplicatedStorage")
local Players = game:GetService("Players")
local remote = RS:WaitForChild("__NAME__WarpRemote")
local gui = Instance.new("ScreenGui")
gui.Name = "__NAME__WarpMenu"
gui.ResetOnSpawn = false
local toggle = Instance.new("TextButton")
toggle.Name = "Toggle"
toggle.Text = "Warp"
toggle.Size = UDim2.fromOffset(120, 44)
toggle.Position = UDim2.new(0, 12, 0.5, -22)
toggle.Parent = gui
local list = Instance.new("Frame")
list.Name = "List"
list.Visible = false
list.Size = UDim2.fromOffset(200, 300)
list.Position = UDim2.new(0, 140, 0.5, -150)
list.Parent = gui
local layout = Instance.new("UIListLayout")
layout.Padding = UDim.new(0, 4)
layout.SortOrder = Enum.SortOrder.Name
layout.Parent = list
local function refresh()
	for _, c in ipairs(list:GetChildren()) do
		if c:IsA("TextButton") then
			c:Destroy()
		end
	end
	local ok, status = pcall(function()
		return remote:InvokeServer("status")
	end)
	if not ok or type(status) ~= "table" then
		return
	end
	for id, s in pairs(status) do
		local b = Instance.new("TextButton")
		b.Name = id
		b.Size = UDim2.new(1, 0, 0, 40)
		b.Text = if s.built then (if s.allowed then id else id .. " (rank " .. tostring(s.need) .. ")") else id .. " (soon)"
		b.BackgroundColor3 = if s.built and s.allowed then Color3.fromRGB(70, 140, 90) else Color3.fromRGB(110, 110, 110)
		b.Parent = list
		b.Activated:Connect(function()
			local warped, why = remote:InvokeServer("warp", id)
			if not warped then
				b.Text = id .. ": " .. tostring(why)
			else
				list.Visible = false
			end
		end)
	end
end
toggle.Activated:Connect(function()
	list.Visible = not list.Visible
	if list.Visible then
		refresh()
	end
end)
gui.Parent = Players.LocalPlayer:WaitForChild("PlayerGui")
'''

WARP_SELFTEST = r'''-- __NAME__WarpSelfTest: the warp template's self-test. Runs only in Studio, only while MifBridge's play probe has put
-- Workspace.MifScratch.SelfTest___NAME__ there; it needs one player with a character. It builds its own islands in a
-- folder of its own, points __NAME__Warps at them for the test, and gives everything back before it reports.
local RunService = game:GetService("RunService")
local HttpService = game:GetService("HttpService")
local Players = game:GetService("Players")
if not RunService:IsStudio() then
	return
end
local scratch = workspace:WaitForChild("MifScratch", 5)
if scratch == nil or scratch:WaitForChild("SelfTest___NAME__", 20) == nil then
	return
end
local Warps = require(game:GetService("ServerScriptService"):WaitForChild("__NAME__Warps"))
local cases = {}
local function case(name, pass, got)
	table.insert(cases, { name = name, pass = pass == true, got = got })
end
local t0 = os.clock()
while #Players:GetPlayers() < 1 and os.clock() - t0 < 20 do
	task.wait(0.2)
end
local player = Players:GetPlayers()[1]
local root = nil
if player then
	local char = player.Character or player.CharacterAdded:Wait()
	root = char:WaitForChild("HumanoidRootPart", 20)
end
if root ~= nil then
	task.wait(2) -- let a character that just spawned settle, so its own first moves are not taken for the warp's
end
local keep = { root = Warps.ROOT_NAME, areas = Warps.AREAS, rankOf = Warps.rankOf, cooldown = Warps.COOLDOWN }
local folder = Instance.new("Folder")
folder.Name = "__NAME__WarpTestIslands"
local function island(id, at)
	local m = Instance.new("Model")
	m.Name = id
	local floor = Instance.new("Part")
	floor.Name = "Floor"
	floor.Anchored, floor.Size, floor.Position = true, Vector3.new(60, 4, 60), at
	floor.Parent = m
	local pad = Instance.new("Part")
	pad.Name = "ArrivalPad"
	pad.Anchored, pad.Size = true, Vector3.new(8, 1, 8)
	pad.CFrame = CFrame.lookAt(at + Vector3.new(0, 2.5, 20), at + Vector3.new(0, 2.5, 0))
	pad.Parent = m
	m.Parent = folder
	return pad
end
local hubPad = island("Hub", Vector3.new(0, 500, 0))
local aPad = island("A", Vector3.new(400, 500, 0))
folder.Parent = scratch
local rank = 0
local ok, err = pcall(function()
	Warps.ROOT_NAME = folder.Name
	Warps.root = function()
		return folder
	end
	Warps.AREAS = { Hub = 0, A = 1, B = 2 }
	Warps.rankOf = function()
		return rank
	end
	Warps.COOLDOWN = 0
	if player == nil or root == nil then
		case("a player with a character joined", false, "no character in 20 s")
		return
	end
	local function near(pad)
		local want = pad.Position + Vector3.new(0, pad.Size.Y / 2 + 3, 0)
		return (root.Position - want).Magnitude < 1, (root.Position - want).Magnitude
	end
	local was = root.Position
	local w1, why1 = Warps.warp(player, "A")
	-- read at once, with no yield between: a refused warp must not have moved it, and nothing else can move it here
	case("a locked warp is refused for rank and the character does not move",
		w1 == false and why1 == "rank" and (root.Position - was).Magnitude < 0.5, { why = why1, moved = (root.Position - was).Magnitude })
	local w2, why2 = Warps.warp(player, "Hub")
	task.wait(1)
	local onHub, dHub = near(hubPad)
	local facing = root.CFrame.LookVector:Dot(Vector3.new(hubPad.CFrame.LookVector.X, 0, hubPad.CFrame.LookVector.Z).Unit)
	case("an allowed warp lands on the arrival pad, facing the area", w2 == true and onHub and facing > 0.99,
		{ why = why2, distance = dHub, facing = facing })
	rank = 1
	local w3 = Warps.warp(player, "A")
	task.wait(1)
	local onA, dA = near(aPad)
	case("the rank that allows it lands on the other island's pad", w3 == true and onA, { distance = dA })
	rank = 5
	local w4, why4 = Warps.warp(player, "B")
	case("a real destination not built yet is refused as unbuilt", w4 == false and why4 == "unbuilt", why4)
	local forged = {}
	for _, id in ipairs({ 123, "Nowhere", "hub", "" }) do
		local wf, whyf = Warps.warp(player, id)
		table.insert(forged, { id = id, warped = wf, why = whyf })
	end
	local wt, whyt = Warps.warp(player, { "A" })
	local wn, whyn = Warps.warp(player, nil)
	local allRefused = wt == false and wn == false and whyt == "not a destination" and whyn == "not a destination"
	for _, f in ipairs(forged) do
		allRefused = allRefused and f.warped == false and f.why == "not a destination"
	end
	case("forged ids (a number, an unknown name, a wrong case, empty, a table, nil) are refused", allRefused, forged)
	Warps.COOLDOWN = 5
	Warps.warp(player, "Hub")
	local w5, why5 = Warps.warp(player, "A")
	Warps.COOLDOWN = 0
	case("a second warp inside the cooldown is refused", w5 == false and why5 == "cooldown", why5)
	local g = Warps.islandGaps()
	case("the island gaps: Hub to A measured from the parts' extents, 340 studs, none too close",
		#g.pairs == 1 and math.abs(g.pairs[1].gap - 340) < 0.01 and #g.tooClose == 0, g)
	island("C", Vector3.new(0, 500, 100))
	local g2 = Warps.islandGaps()
	local close = {}
	for _, r in ipairs(g2.tooClose) do
		table.insert(close, r.a .. "-" .. r.b)
	end
	case("an island built 40 studs from the hub is named too close", table.find(close, "C-Hub") ~= nil, g2.tooClose)
end)
Warps.ROOT_NAME, Warps.AREAS, Warps.rankOf, Warps.COOLDOWN = keep.root, keep.areas, keep.rankOf, keep.cooldown
Warps.root = function()
	return workspace:FindFirstChild(Warps.ROOT_NAME)
end
folder:Destroy()
if not ok then
	case("the self-test ran to the end", false, tostring(err))
end
local passed = 0
for _, c in ipairs(cases) do
	if c.pass then
		passed += 1
	end
end
local j = HttpService:JSONEncode({ template = "warps", name = "__NAME__", passed = passed, total = #cases, cases = cases })
--@emit MIFSELFTEST __NAME__.warps
'''


# ---------------------------------------------------------------- template "subscription": an entitlement Roblox answers
SUB_MODULE = r'''-- __NAME__Subscription: who holds the subscription, as Roblox answers it. Written by MifBridge's rbx_script_scaffold
-- (kind subscription). The entitlement is Roblox's answer and nothing else: read when a player joins, when Roblox says
-- a status changed, after a purchase prompt closes, and on a timer. It is held for the session only and never goes
-- into a save, so a lapsed, cancelled or refunded subscription ends at the next read. A failed read is not subscribed
-- and is read again after RETRY seconds. A prompt's own "finished" event grants nothing (the same rule as for
-- game passes): it only asks Roblox again. Nothing here creates or prices a subscription.
-- Read from Studio 0.748's API dump: MarketplaceService:GetUserSubscriptionStatusAsync(user: Player, subscriptionId:
-- string) answers a Dictionary; :PromptSubscriptionPurchase(user: Player, subscriptionId: string);
-- .PromptSubscriptionPurchaseFinished(user, subscriptionId, didTryPurchasing); Players.UserSubscriptionStatusChanged(
-- user, subscriptionId).
-- ASSUMED from Roblox's documentation, not measured: the status Dictionary's field is IsSubscribed (a boolean); the
-- status read and the prompt are both called from the server; the id is the string Creator Hub shows for the
-- subscription. Check each on a published place before selling anything on it.

local Sub = {}
Sub.SUBSCRIPTION_ID = ""  -- the subscription's id from Creator Hub; "" leaves the template off
Sub.RECHECK = 300         -- seconds between timed reads of one player's status
Sub.RETRY = 30            -- seconds before a failed read is tried again
Sub.PROMPT_GAP = 10       -- seconds between one player's prompts
Sub.market = function() return game:GetService("MarketplaceService") end
Sub.clock = function() return os.clock() end

local held = {}      -- [player] = { subscribed, failed, readAt, promptedAt }: this session only, never saved
local listeners = {} -- function(player, subscribed), called when a player's entitlement changes

function Sub.configured()
	return type(Sub.SUBSCRIPTION_ID) == "string" and Sub.SUBSCRIPTION_ID ~= ""
end

local function set(player, h, subscribed)
	if h.subscribed == subscribed then return end
	h.subscribed = subscribed
	for _, fn in ipairs(listeners) do
		task.spawn(fn, player, subscribed)
	end
end

-- Ask Roblox. Answers subscribed, and whether the read worked; a failed read is not subscribed.
function Sub.read(player)
	local h = held[player]
	if h == nil then return false, false end
	if not Sub.configured() then
		h.readAt = Sub.clock()
		return false, false
	end
	local ok, status = pcall(function()
		return Sub.market():GetUserSubscriptionStatusAsync(player, Sub.SUBSCRIPTION_ID)
	end)
	if held[player] ~= h then return false, ok end -- left during the read
	h.readAt = Sub.clock()
	h.failed = not ok
	set(player, h, ok and type(status) == "table" and status.IsSubscribed == true)
	return h.subscribed, ok
end

function Sub.join(player)
	held[player] = { subscribed = false, failed = false, readAt = 0, promptedAt = -math.huge }
	return Sub.read(player)
end

function Sub.leave(player)
	held[player] = nil
end

-- Is this player entitled now? Ask this on the server wherever the subscription gives something.
function Sub.has(player)
	local h = held[player]
	return h ~= nil and h.subscribed == true
end

function Sub.onChange(fn)
	table.insert(listeners, fn)
end

-- Roblox said a status changed, or a purchase prompt closed: read again. The event itself grants nothing.
function Sub.changed(player, subscriptionId)
	if subscriptionId ~= Sub.SUBSCRIPTION_ID or held[player] == nil then return end
	Sub.read(player)
end

-- The timer: every player whose last read is RECHECK old, or RETRY old after a failed read. Answers how many it read.
function Sub.recheckOnce()
	local now = Sub.clock()
	local due = {}
	for player, h in pairs(held) do
		if now - h.readAt >= (if h.failed then Sub.RETRY else Sub.RECHECK) then
			table.insert(due, player)
		end
	end
	for _, player in ipairs(due) do
		task.spawn(Sub.read, player)
	end
	return #due
end

-- The client asked to subscribe. The status is read first: a subscriber is never prompted, and a failed read prompts
-- nothing. Answers true when the prompt was sent, else false and why.
function Sub.prompt(player)
	local h = held[player]
	if h == nil then return false, "not joined" end
	if not Sub.configured() then return false, "no subscription id set" end
	local now = Sub.clock()
	if now - h.promptedAt < Sub.PROMPT_GAP then return false, "too soon" end
	h.promptedAt = now
	local subscribed, ok = Sub.read(player)
	if not ok then return false, "status read failed" end
	if subscribed then return false, "already subscribed" end
	local sent = pcall(function()
		Sub.market():PromptSubscriptionPurchase(player, Sub.SUBSCRIPTION_ID)
	end)
	if not sent then return false, "the prompt failed" end
	return true
end

return Sub
'''

SUB_SERVER = r'''-- __NAME__SubscriptionServer: wires __NAME__Subscription to players, Roblox's change events, the timer and the one
-- remote. Written by MifBridge's rbx_script_scaffold. The player's __NAME__Subscribed attribute mirrors the answer for
-- the client's interface only; server code asks Sub.has(player).
local Players = game:GetService("Players")
local MarketplaceService = game:GetService("MarketplaceService")
local Sub = require(game:GetService("ServerScriptService"):WaitForChild("__NAME__Subscription"))
local remote = game:GetService("ReplicatedStorage"):WaitForChild("__NAME__SubscriptionRemote")

Sub.onChange(function(player, subscribed)
	if typeof(player) == "Instance" and player.Parent then
		player:SetAttribute("__NAME__Subscribed", subscribed)
	end
end)

local function join(player)
	if Sub.configured() then
		player:SetAttribute("__NAME__Subscribed", false)
	end
	Sub.join(player)
end

Players.PlayerAdded:Connect(join)
for _, player in ipairs(Players:GetPlayers()) do
	task.spawn(join, player)
end
Players.PlayerRemoving:Connect(Sub.leave)
Players.UserSubscriptionStatusChanged:Connect(Sub.changed)
MarketplaceService.PromptSubscriptionPurchaseFinished:Connect(function(player, subscriptionId)
	Sub.changed(player, subscriptionId)
end)
remote.OnServerInvoke = function(player, what)
	if what == "prompt" then
		return Sub.prompt(player)
	end
	return Sub.has(player)
end
task.spawn(function()
	while true do
		task.wait(5)
		Sub.recheckOnce()
	end
end)
'''

SUB_CLIENT = r'''-- __NAME__SubscriptionClient: a subscribe button, shown while the server says this player is not subscribed. Written
-- by MifBridge's rbx_script_scaffold (kind subscription). The client only asks: the server reads the status from
-- Roblox and sends the prompt. No button shows while the template is off (no id set). Restyle the button; keep the call.
local Players = game:GetService("Players")
local player = Players.LocalPlayer
local remote = game:GetService("ReplicatedStorage"):WaitForChild("__NAME__SubscriptionRemote")

local gui = Instance.new("ScreenGui")
gui.Name = "__NAME__SubscriptionGui"
gui.ResetOnSpawn = false
gui.Enabled = false
local button = Instance.new("TextButton")
button.Name = "Subscribe"
button.AnchorPoint = Vector2.new(1, 1)
button.Position = UDim2.new(1, -16, 1, -68)
button.Size = UDim2.fromOffset(220, 44)
button.Text = "Subscribe"
button.Parent = gui
gui.Parent = player:WaitForChild("PlayerGui")

local function refresh()
	gui.Enabled = player:GetAttribute("__NAME__Subscribed") == false
end
player:GetAttributeChangedSignal("__NAME__Subscribed"):Connect(refresh)
refresh()
local busy = false
button.Activated:Connect(function()
	if busy then return end
	busy = true
	pcall(function() return remote:InvokeServer("prompt") end)
	task.wait(2)
	busy = false
end)
'''

# Self-test of the subscription template: fake players and a recorder in place of the marketplace calls, with its own
# clock, so no case waits and nothing is asked of Roblox.
SUB_SELFTEST = r'''-- __NAME__SubscriptionSelfTest: the subscription template's self-test. Runs only in Studio, only while MifBridge's
-- play probe has put Workspace.MifScratch.SelfTest___NAME__ there; otherwise it does nothing. A recorder stands in for
-- the marketplace calls and the players are tables, so nothing is asked of Roblox and nobody is prompted.
local RunService = game:GetService("RunService")
local HttpService = game:GetService("HttpService")
if not RunService:IsStudio() then return end
local scratch = workspace:WaitForChild("MifScratch", 5)
if scratch == nil or scratch:WaitForChild("SelfTest___NAME__", 20) == nil then return end
local Sub = require(game:GetService("ServerScriptService"):WaitForChild("__NAME__Subscription"))
local cases = {}
local function case(name, fn)
	local ok, pass, got = pcall(fn)
	table.insert(cases, { name = name, pass = ok and pass == true, got = if ok then got else "error: " .. tostring(pass) })
end
local function recorder()
	local rec = { reads = {}, prompts = {}, status = {}, fail = false }
	function rec:GetUserSubscriptionStatusAsync(player, id)
		rec.reads[player] = (rec.reads[player] or 0) + 1
		if rec.fail then error("recorder: the status read failed") end
		return { IsSubscribed = rec.status[player] == true }
	end
	function rec:PromptSubscriptionPurchase(player, id)
		table.insert(rec.prompts, { player = player, id = id })
	end
	return rec
end
local ID = "SELFTEST-SUBSCRIPTION"
local keep = { id = Sub.SUBSCRIPTION_ID, market = Sub.market, clock = Sub.clock }
local rec = recorder()
local now = 1000
Sub.market = function() return rec end
Sub.clock = function() return now end
local P0, P1, P2, P3 = { Name = "P0" }, { Name = "P1" }, { Name = "P2" }, { Name = "P3" }
local function reads(p)
	return rec.reads[p] or 0
end
local told = {}
local listening = true
Sub.onChange(function(player, subscribed)
	if listening then table.insert(told, { player = player, subscribed = subscribed }) end
end)

case("off until the subscription id is set: nothing is read and nothing is prompted", function()
	Sub.SUBSCRIPTION_ID = ""
	rec.status[P0] = true
	local subscribed = Sub.join(P0)
	local sent, why = Sub.prompt(P0)
	Sub.leave(P0)
	Sub.SUBSCRIPTION_ID = ID
	return subscribed == false and sent == false and why == "no subscription id set" and reads(P0) == 0 and #rec.prompts == 0,
		{ subscribed = subscribed, why = why, reads = reads(P0), prompts = #rec.prompts }
end)

case("a subscriber is entitled from the read at join", function()
	rec.status[P1] = true
	local before = Sub.has(P1)
	Sub.join(P1)
	return before == false and Sub.has(P1) == true and reads(P1) == 1, { before = before, has = Sub.has(P1), reads = reads(P1) }
end)

case("a player who is not subscribed is not entitled", function()
	Sub.join(P2)
	return Sub.has(P2) == false and reads(P2) == 1, { has = Sub.has(P2), reads = reads(P2) }
end)

case("a lapsed subscription ends when Roblox says it changed; another subscription's change reads nothing", function()
	rec.status[P1] = false
	Sub.changed(P1, "ANOTHER-SUBSCRIPTION")
	local still, readsOther = Sub.has(P1), reads(P1)
	Sub.changed(P1, ID)
	return still == true and readsOther == 1 and Sub.has(P1) == false and reads(P1) == 2,
		{ afterOther = still, has = Sub.has(P1), reads = reads(P1) }
end)

case("the timer reads again after RECHECK and not before, and finds a subscription bought elsewhere", function()
	rec.status[P2] = true
	local before = reads(P2)
	now += Sub.RECHECK - 1
	Sub.recheckOnce()
	local early, hadEarly = reads(P2) - before, Sub.has(P2)
	now += 1
	Sub.recheckOnce()
	return early == 0 and hadEarly == false and reads(P2) - before == 1 and Sub.has(P2) == true,
		{ early = early, reads = reads(P2) - before, has = Sub.has(P2) }
end)

case("a failed read is not subscribed, and is read again after RETRY", function()
	local before = reads(P2)
	rec.fail = true
	now += Sub.RECHECK
	Sub.recheckOnce()
	local failedHas, failedReads = Sub.has(P2), reads(P2) - before
	rec.fail = false
	now += Sub.RETRY - 1
	Sub.recheckOnce()
	local early = reads(P2) - before
	now += 1
	Sub.recheckOnce()
	return failedHas == false and failedReads == 1 and early == 1 and reads(P2) - before == 2 and Sub.has(P2) == true,
		{ afterFailure = failedHas, early = early, reads = reads(P2) - before, has = Sub.has(P2) }
end)

case("a subscriber is never prompted; a failed read prompts nothing; one prompt per PROMPT_GAP", function()
	Sub.join(P3)
	local a, whyA = Sub.prompt(P2) -- subscribed
	rec.fail = true
	local b, whyB = Sub.prompt(P3)
	rec.fail = false
	local c, whyC = Sub.prompt(P3) -- inside the gap
	now += Sub.PROMPT_GAP
	local d = Sub.prompt(P3)
	local sent = rec.prompts[1]
	return a == false and whyA == "already subscribed" and b == false and whyB == "status read failed" and c == false
		and whyC == "too soon" and d == true and #rec.prompts == 1 and sent.player == P3 and sent.id == ID
		and Sub.has(P3) == false, { subscriber = whyA, failed = whyB, gap = whyC, sent = d, prompts = #rec.prompts }
end)

case("a change is told to listeners once", function()
	local n = 0
	for _, t in ipairs(told) do
		if t.player == P2 then n += 1 end
	end
	-- P2: subscribed by the timer, dropped by the failed read, subscribed again by the retry; a repeated answer tells nobody
	local last = told[#told]
	Sub.changed(P2, ID)
	return n == 3 and told[#told] == last, { toldP2 = n, total = #told }
end)

case("a leave forgets the entitlement and a rejoin asks Roblox again", function()
	local before = reads(P2)
	Sub.leave(P2)
	local gone = Sub.has(P2)
	rec.status[P2] = false
	Sub.join(P2)
	return gone == false and Sub.has(P2) == false and reads(P2) - before == 1, { gone = gone, has = Sub.has(P2), reads = reads(P2) - before }
end)

listening = false
for _, p in ipairs({ P0, P1, P2, P3 }) do
	Sub.leave(p)
end
Sub.SUBSCRIPTION_ID, Sub.market, Sub.clock = keep.id, keep.market, keep.clock
case("the self-test ran to the end", function()
	return Sub.SUBSCRIPTION_ID == keep.id and Sub.has(P1) == false, Sub.SUBSCRIPTION_ID
end)

local passed = 0
for _, c in ipairs(cases) do
	if c.pass then passed += 1 end
end
local j = HttpService:JSONEncode({ template = "subscription", name = "__NAME__", passed = passed, total = #cases, cases = cases })
--@emit MIFSELFTEST __NAME__.subscription
'''


# ---------------------------------------------------------------- template "analytics"
ANALYTICS_MODULE = r'''-- __NAME__Analytics: the one module that sends this game's analytics events. Written by MifBridge's
-- rbx_script_scaffold (kind analytics). Every name it sends is declared in a table below, so no name is ever built from
-- player data; an economy event is sent only after the change it reports landed; an onboarding step is sent once, in
-- order; and every call is counted against a budget. Call it from server scripts only. Nothing here yields.
--
-- ASSUMED, not measured by MifBridge: the limits in the comments below (100 steps a funnel, 10 currency types, 100
-- custom event names, 3 custom fields, 120 + 20 x players requests a minute) are Roblox's documented ones as read on
-- 2026-11-22. rbx_analytics_audit counts a game's scripts against the same numbers.
local Players = game:GetService("Players")

local Analytics = {}
Analytics.service = game:GetService("AnalyticsService") -- the self-test puts a recorder here
Analytics.now = os.clock
Analytics.players = function()
	return #Players:GetPlayers()
end
-- The onboarding funnel: what a new player meets, in order. A step's number is its place in this list (at most 100).
Analytics.ONBOARDING = { "Joined", "FirstAction", "FirstReward", "FirstUpgrade" }
-- Currencies, key = the name sent (at most 10). Change a currency only through Analytics.grant and Analytics.spend.
Analytics.CURRENCIES = { Coins = "Coins" }
-- Custom events, key = the name sent (at most 100). Add a line here before sending a new event.
Analytics.EVENTS = { RoundPlayed = "RoundPlayed", ItemEquipped = "ItemEquipped" }
Analytics.BUDGET_BASE = 120      -- requests a window for the server
Analytics.BUDGET_PER_PLAYER = 20 -- plus this many for each player in it
Analytics.WINDOW = 60            -- seconds
Analytics.dropped = 0            -- events not sent because the window's budget was spent

local reached = {} -- player -> the highest onboarding step sent
local windowStart, used = nil, 0

local function budget()
	local t = Analytics.now()
	if windowStart == nil or t - windowStart >= Analytics.WINDOW then
		windowStart, used = t, 0
	end
	if used >= Analytics.BUDGET_BASE + Analytics.BUDGET_PER_PLAYER * Analytics.players() then
		Analytics.dropped += 1
		return false
	end
	used += 1
	return true
end

local FIELD_KEYS = { Enum.AnalyticsCustomFieldKeys.CustomField01, Enum.AnalyticsCustomFieldKeys.CustomField02,
	Enum.AnalyticsCustomFieldKeys.CustomField03 }
-- A list of up to 3 values -> the customFields table keyed CustomField01..03. Answers the table (or nil) and whether
-- the list was fine.
local function fieldsOf(list)
	if list == nil then
		return nil, true
	end
	if type(list) ~= "table" or #list > #FIELD_KEYS then
		return nil, false
	end
	local out = {}
	for i, v in ipairs(list) do
		out[FIELD_KEYS[i].Name] = tostring(v)
	end
	return out, true
end

-- An onboarding step, by its name in ONBOARDING. Sent once per player, and only as the next step: a repeat, a step out
-- of order and an undeclared name are refused. Answers sent (bool) and why.
function Analytics.step(player, stepName)
	local index = table.find(Analytics.ONBOARDING, stepName)
	if index == nil then
		return false, "not a declared step"
	end
	local at = reached[player] or 0
	if index <= at then
		return false, "already sent"
	end
	if index ~= at + 1 then
		return false, "out of order: " .. tostring(Analytics.ONBOARDING[at + 1]) .. " comes first"
	end
	if not budget() then
		return false, "over the budget"
	end
	local ok, err = pcall(function()
		Analytics.service:LogOnboardingFunnelStepEvent(player, index, Analytics.ONBOARDING[index])
	end)
	if not ok then
		return false, "the call failed: " .. tostring(err)
	end
	reached[player] = index
	return true, "sent"
end

-- Forget a player who left.
function Analytics.forget(player)
	reached[player] = nil
end

local function flow(flowType, player, currencyKey, amount, transaction, itemSku, change)
	local currency = Analytics.CURRENCIES[currencyKey]
	if currency == nil then
		return false, "not a declared currency"
	end
	if type(amount) ~= "number" or amount ~= amount or amount <= 0 then
		return false, "amount is a number above 0"
	end
	if typeof(transaction) ~= "EnumItem" or transaction.EnumType ~= Enum.AnalyticsEconomyTransactionType then
		return false, "transaction is an Enum.AnalyticsEconomyTransactionType item"
	end
	if type(change) ~= "function" then
		return false, "change is the function that moves the balance"
	end
	-- the change first: the event reports what landed, never the attempt
	local landed, balance = pcall(change, amount)
	if not landed or type(balance) ~= "number" then
		return false, "the change did not land"
	end
	if not budget() then
		return true, "applied; not sent: over the budget", balance
	end
	local ok, err = pcall(function()
		Analytics.service:LogEconomyEvent(player, flowType, currency, amount, balance, transaction.Name, itemSku or "")
	end)
	if not ok then
		return true, "applied; the call failed: " .. tostring(err), balance
	end
	return true, "sent", balance
end

-- The currency's one grant function. change(amount) adds it (and saves, when the game saves here) and returns the new
-- balance; nil or an error means it did not land, and then nothing is sent. transaction is an
-- Enum.AnalyticsEconomyTransactionType item. Answers applied (bool), why, and the new balance.
function Analytics.grant(player, currencyKey, amount, transaction, itemSku, change)
	return flow(Enum.AnalyticsEconomyFlowType.Source, player, currencyKey, amount, transaction, itemSku, change)
end

-- The currency's one spend function: the same, with change(amount) taking it away (and returning nil when the player
-- cannot pay).
function Analytics.spend(player, currencyKey, amount, transaction, itemSku, change)
	return flow(Enum.AnalyticsEconomyFlowType.Sink, player, currencyKey, amount, transaction, itemSku, change)
end

-- A custom event, by its key in EVENTS, with a number (default 1) and up to 3 custom field values as a list.
-- Answers sent (bool) and why.
function Analytics.event(player, key, value, fields)
	local name = Analytics.EVENTS[key]
	if name == nil then
		return false, "not a declared event"
	end
	if value ~= nil and (type(value) ~= "number" or value ~= value) then
		return false, "value is a number"
	end
	local custom, fine = fieldsOf(fields)
	if not fine then
		return false, "fields is a list of at most 3 values"
	end
	if not budget() then
		return false, "over the budget"
	end
	local ok, err = pcall(function()
		Analytics.service:LogCustomEvent(player, name, value or 1, custom)
	end)
	if not ok then
		return false, "the call failed: " .. tostring(err)
	end
	return true, "sent"
end

-- Start the budget's window again (the self-test).
function Analytics.reset()
	windowStart, used = nil, 0
	Analytics.dropped = 0
end

return Analytics
'''

ANALYTICS_SERVER = r'''-- __NAME__AnalyticsServer: sends the first onboarding step when a player joins and forgets a player who left.
-- Written by MifBridge's rbx_script_scaffold (kind analytics). Send the later steps where the player reaches them:
--   Analytics.step(player, "FirstAction")
-- and move a currency only through the module, so each source and sink is sent after it landed:
--   Analytics.grant(player, "Coins", 25, Enum.AnalyticsEconomyTransactionType.Gameplay, "round_win", function(amount)
--       profile.Data.Coins += amount
--       return profile.Data.Coins
--   end)
local Players = game:GetService("Players")
local Analytics = require(game:GetService("ServerScriptService"):WaitForChild("__NAME__Analytics"))

local function joined(player)
	Analytics.step(player, Analytics.ONBOARDING[1])
end
Players.PlayerAdded:Connect(joined)
for _, p in ipairs(Players:GetPlayers()) do
	joined(p)
end
Players.PlayerRemoving:Connect(Analytics.forget)
'''

ANALYTICS_SELFTEST = r'''-- __NAME__AnalyticsSelfTest: the analytics template's self-test. Runs only in Studio, only while MifBridge's play probe
-- has put Workspace.MifScratch.SelfTest___NAME__ there. A recorder stands in place of AnalyticsService, so nothing is
-- sent to Roblox; the module's own tables are used, and everything is given back before it reports.
local RunService = game:GetService("RunService")
local HttpService = game:GetService("HttpService")
if not RunService:IsStudio() then
	return
end
local scratch = workspace:WaitForChild("MifScratch", 5)
if scratch == nil or scratch:WaitForChild("SelfTest___NAME__", 20) == nil then
	return
end
local Analytics = require(game:GetService("ServerScriptService"):WaitForChild("__NAME__Analytics"))
local cases = {}
local function case(name, pass, got)
	table.insert(cases, { name = name, pass = pass == true, got = got })
end
local sent = {}
local refuse = false
local function record(member, a, b, c, d, e, f, g)
	if refuse then
		refuse = false
		error("the recorder refused this call")
	end
	table.insert(sent, { member = member, a = a, b = b, c = c, d = d, e = e, f = f, g = g })
end
local recorder = {}
function recorder:LogOnboardingFunnelStepEvent(_player, step, stepName, customFields)
	record("LogOnboardingFunnelStepEvent", step, stepName, customFields)
end
function recorder:LogEconomyEvent(_player, flowType, currencyType, amount, endingBalance, transactionType, itemSku, customFields)
	record("LogEconomyEvent", flowType.Name, currencyType, amount, endingBalance, transactionType, itemSku, customFields)
end
function recorder:LogCustomEvent(_player, eventName, value, customFields)
	record("LogCustomEvent", eventName, value, customFields)
end
local real = { service = Analytics.service, now = Analytics.now, players = Analytics.players }
local clock, online = 1000, 1
Analytics.service = recorder
Analytics.now = function()
	return clock
end
Analytics.players = function()
	return online
end
Analytics.reset()
local player = { UserId = 1 }
local steps = Analytics.ONBOARDING
local curKey = next(Analytics.CURRENCIES)
local evKey = next(Analytics.EVENTS)
local T = Enum.AnalyticsEconomyTransactionType
local F = Enum.AnalyticsCustomFieldKeys
if #steps >= 2 and curKey ~= nil and evKey ~= nil then
	-- the onboarding funnel
	local early, earlyWhy = Analytics.step(player, steps[2])
	case("a later step before the first is refused", early == false and #sent == 0, { ok = early, why = earlyWhy, sent = #sent })
	local first = Analytics.step(player, steps[1])
	local ev = sent[1] or {}
	case("the first step is sent with its number and its name",
		first == true and #sent == 1 and ev.member == "LogOnboardingFunnelStepEvent" and ev.a == 1 and ev.b == steps[1], ev)
	local again = Analytics.step(player, steps[1])
	case("a step is sent once", again == false and #sent == 1, { ok = again, sent = #sent })
	local unknown = Analytics.step(player, "NotDeclared")
	case("an undeclared step is refused", unknown == false and #sent == 1, { ok = unknown, sent = #sent })
	refuse = true
	local failed = Analytics.step(player, steps[2])
	local retried = Analytics.step(player, steps[2])
	ev = sent[2] or {}
	case("a step whose call failed is not marked sent, and goes the next time",
		failed == false and retried == true and #sent == 2 and ev.a == 2, { failed = failed, retried = retried, sent = #sent })
	Analytics.forget(player)
	local fresh = Analytics.step(player, steps[1])
	case("forget starts a player's funnel again", fresh == true and #sent == 3, { ok = fresh, sent = #sent })
	-- economy: the change first, the event after it landed
	local wallet = 10
	local function add(amount)
		wallet += amount
		return wallet
	end
	local function take(amount)
		if wallet < amount then
			return nil
		end
		wallet -= amount
		return wallet
	end
	local n0 = #sent
	local g, gWhy, gBalance = Analytics.grant(player, curKey, 5, T.Gameplay, "selftest", add)
	ev = sent[n0 + 1] or {}
	case("a grant applies the change, then sends a Source event with the ending balance",
		g == true and gBalance == 15 and wallet == 15 and #sent == n0 + 1 and ev.member == "LogEconomyEvent"
			and ev.a == "Source" and ev.b == Analytics.CURRENCIES[curKey] and ev.c == 5 and ev.d == 15
			and ev.e == T.Gameplay.Name and ev.f == "selftest", { ok = g, why = gWhy, event = ev })
	local s, sWhy, sBalance = Analytics.spend(player, curKey, 4, T.Shop, "selftest", take)
	ev = sent[n0 + 2] or {}
	case("a spend sends a Sink event",
		s == true and sBalance == 11 and wallet == 11 and ev.a == "Sink" and ev.c == 4 and ev.d == 11 and ev.e == T.Shop.Name,
		{ ok = s, why = sWhy, event = ev })
	local n1 = #sent
	local poor, poorWhy = Analytics.spend(player, curKey, 999, T.Shop, "selftest", take)
	case("a change that did not land sends nothing", poor == false and wallet == 11 and #sent == n1,
		{ ok = poor, why = poorWhy, wallet = wallet, sent = #sent - n1 })
	local threw, threwWhy = Analytics.grant(player, curKey, 1, T.Gameplay, "selftest", function()
		error("the save failed")
	end)
	case("a change that errors sends nothing", threw == false and #sent == n1, { ok = threw, why = threwWhy, sent = #sent - n1 })
	local ran = false
	local function mark()
		ran = true
		return 1
	end
	local undeclared = Analytics.grant(player, "NotDeclared", 1, T.Gameplay, "selftest", mark)
	local badType = Analytics.grant(player, curKey, 1, "Shop", "selftest", mark)
	local badAmount = Analytics.grant(player, curKey, -1, T.Gameplay, "selftest", mark)
	case("an undeclared currency, a transaction that is no enum item and an amount of 0 or less are refused before the change runs",
		undeclared == false and badType == false and badAmount == false and ran == false and #sent == n1,
		{ undeclared = undeclared, badType = badType, badAmount = badAmount, ran = ran, sent = #sent - n1 })
	-- custom events
	local c1 = Analytics.event(player, evKey, 3, { "a", "b" })
	ev = sent[n1 + 1] or {}
	case("a declared event is sent by its declared name with its value and its fields",
		c1 == true and ev.member == "LogCustomEvent" and ev.a == Analytics.EVENTS[evKey] and ev.b == 3 and type(ev.c) == "table"
			and ev.c[F.CustomField01.Name] == "a" and ev.c[F.CustomField02.Name] == "b", { ok = c1, event = ev })
	local c2 = Analytics.event(player, "NotDeclared", 1)
	local c3 = Analytics.event(player, evKey, 1, { "a", "b", "c", "d" })
	case("an undeclared event and a fourth custom field are refused", c2 == false and c3 == false and #sent == n1 + 1,
		{ undeclared = c2, fourFields = c3, sent = #sent - n1 })
	-- the budget
	Analytics.reset()
	online = 2
	local limit = Analytics.BUDGET_BASE + Analytics.BUDGET_PER_PLAYER * online
	local n2 = #sent
	local went = 0
	for _ = 1, limit + 5 do
		if Analytics.event(player, evKey, 1) then
			went += 1
		end
	end
	case("a window sends BUDGET_BASE + BUDGET_PER_PLAYER x players; the rest are dropped and counted",
		went == limit and #sent == n2 + limit and Analytics.dropped == 5, { went = went, limit = limit, dropped = Analytics.dropped })
	local before = wallet
	local over, overWhy = Analytics.grant(player, curKey, 1, T.Gameplay, "selftest", add)
	case("over the budget a grant still applies its change, and says it was not sent",
		over == true and wallet == before + 1 and #sent == n2 + limit and Analytics.dropped == 6,
		{ ok = over, why = overWhy, wallet = wallet, dropped = Analytics.dropped })
	clock += Analytics.WINDOW
	local later = Analytics.event(player, evKey, 1)
	case("the next window sends again", later == true and #sent == n2 + limit + 1, { ok = later, sent = #sent - n2 - limit })
else
	case("ONBOARDING has two steps or more, CURRENCIES and EVENTS an entry each", false,
		{ steps = #steps, currency = curKey ~= nil, event = evKey ~= nil })
end
Analytics.forget(player)
Analytics.service, Analytics.now, Analytics.players = real.service, real.now, real.players
Analytics.reset()
local passed = 0
for _, c in ipairs(cases) do
	if c.pass then
		passed += 1
	end
end
local j = HttpService:JSONEncode({ template = "analytics", name = "__NAME__", passed = passed, total = #cases, cases = cases })
--@emit MIFSELFTEST __NAME__.analytics
'''


# ---------------------------------------------------------------- kind filtered_text (rbx_text_filter_audit's template)
FILTER_MODULE = r'''-- __NAME__Text: player-written text (a pet name, a sign, a typed answer), filtered on the server before anyone else
-- sees it or it is saved. Written by MifBridge's rbx_script_scaffold (kind filtered_text). Roblox makes the game
-- responsible for filtering any text it shows that it did not write; this module is the one place that does it:
--   Text.submit(player, text) -> true, the filtered text | false, why
-- It filters with the AUTHOR's user id (TextService:FilterStringAsync(text, author.UserId)) and shows the result to
-- everyone (GetNonChatStringForBroadcastAsync); for text only one other player sees, use GetNonChatStringForUserAsync.
-- A filter that fails or errors shows nothing: the text is refused, never shown raw. In Studio the filter returns the
-- text unchanged (ledger RB3083), so only the self-test's stand-in shows the masking.
local TextService = game:GetService("TextService")

local Text = {}
Text.MAX = 100           -- characters a player may submit
Text.COOLDOWN = 1        -- seconds between two submits from one player
Text.service = TextService
Text.now = os.clock
-- what the game does with filtered text: replace with your sign, pet name or saved field
Text.apply = function(player, shown)
	player:SetAttribute("__NAME__Text", shown)
end

local last = {}

function Text.submit(player, text)
	if type(text) ~= "string" then
		return false, "not text"
	end
	if #text == 0 or #text > Text.MAX or utf8.len(text) == nil then
		return false, "bad length"
	end
	local t = Text.now()
	if last[player] and t - last[player] < Text.COOLDOWN then
		return false, "too fast"
	end
	last[player] = t
	local ok, result = pcall(function()
		return Text.service:FilterStringAsync(text, player.UserId)
	end)
	if not ok or result == nil then
		return false, "filter failed"
	end
	local ok2, shown = pcall(function()
		return result:GetNonChatStringForBroadcastAsync()
	end)
	if not ok2 or type(shown) ~= "string" then
		return false, "filter failed"
	end
	Text.apply(player, shown)
	return true, shown
end

function Text.forget(player)
	last[player] = nil
end

return Text
'''

FILTER_SERVER = r'''-- __NAME__TextServer: a client's text reaches the game only through __NAME__Text.submit, which filters it first.
-- Written by MifBridge's rbx_script_scaffold (kind filtered_text).
local Players = game:GetService("Players")
local RS = game:GetService("ReplicatedStorage")
local Text = require(game:GetService("ServerScriptService"):WaitForChild("__NAME__Text"))
local remote = RS:WaitForChild("__NAME__TextRemote")

remote.OnServerEvent:Connect(function(player, text)
	Text.submit(player, text)
end)
Players.PlayerRemoving:Connect(Text.forget)
'''

FILTER_CLIENT = r'''-- __NAME__TextClient: sends a tagged TextBox's text once, when the player presses Enter - never per keystroke, and
-- never filtered here (a client's filter is no filter: the server filters it). Tag the TextBox "__NAME__Input".
-- Written by MifBridge's rbx_script_scaffold (kind filtered_text).
local CollectionService = game:GetService("CollectionService")
local RS = game:GetService("ReplicatedStorage")
local remote = RS:WaitForChild("__NAME__TextRemote")

local function hook(box)
	if not box:IsA("TextBox") then
		return
	end
	box.FocusLost:Connect(function(enterPressed)
		if enterPressed and #box.Text > 0 then
			remote:FireServer(box.Text)
		end
	end)
end
for _, box in ipairs(CollectionService:GetTagged("__NAME__Input")) do
	hook(box)
end
CollectionService:GetInstanceAddedSignal("__NAME__Input"):Connect(hook)
'''

FILTER_SELFTEST = r'''-- __NAME__TextSelfTest: the filtered-text template's self-test. Runs only in Studio, only while MifBridge's play probe
-- has put Workspace.MifScratch.SelfTest___NAME__ there. A stand-in takes TextService's place (Studio's own filter returns
-- the text unchanged, ledger RB3083), so the masking is the stand-in's; everything is given back before it reports.
local RunService = game:GetService("RunService")
local HttpService = game:GetService("HttpService")
if not RunService:IsStudio() then
	return
end
local scratch = workspace:WaitForChild("MifScratch", 5)
if scratch == nil or scratch:WaitForChild("SelfTest___NAME__", 20) == nil then
	return
end
local Text = require(game:GetService("ServerScriptService"):WaitForChild("__NAME__Text"))
local cases = {}
local function case(name, pass, got)
	table.insert(cases, { name = name, pass = pass == true, got = got })
end
local calls, breakNext = {}, false
local standIn = {}
function standIn:FilterStringAsync(text, userId)
	table.insert(calls, { text = text, userId = userId })
	if breakNext then
		breakNext = false
		error("the filter is down")
	end
	local result = {}
	function result:GetNonChatStringForBroadcastAsync()
		return (string.gsub(text, "bad", "###"))
	end
	return result
end
local real = { service = Text.service, now = Text.now, apply = Text.apply }
local clock = 1000
local applied = {}
Text.service = standIn
Text.now = function()
	return clock
end
Text.apply = function(player, shown)
	table.insert(applied, { userId = player.UserId, shown = shown })
end
local author = { UserId = 4242 }
local function step()
	clock += Text.COOLDOWN + 0.01
end

step()
local ok, shown = Text.submit(author, "a bad word")
case("text is filtered with the author's user id and the broadcast string is what the game gets",
	ok == true and shown == "a ### word" and #calls == 1 and calls[1].userId == 4242 and #applied == 1
		and applied[1].shown == "a ### word",
	{ ok = ok, shown = shown, calls = #calls, applied = applied[1] })
local n0 = #calls
step()
local okN, whyN = Text.submit(author, 12345)
case("a number is refused before the filter", okN == false and whyN == "not text" and #calls == n0, { ok = okN, why = whyN })
step()
local okE, whyE = Text.submit(author, "")
case("an empty string is refused", okE == false and whyE == "bad length" and #calls == n0, { ok = okE, why = whyE })
step()
local okL, whyL = Text.submit(author, string.rep("x", Text.MAX + 1))
case("text over MAX is refused", okL == false and whyL == "bad length" and #calls == n0, { ok = okL, why = whyL })
step()
breakNext = true
local nA = #applied
local okF, whyF = Text.submit(author, "hello")
case("a filter that errors shows nothing: refused, nothing applied",
	okF == false and whyF == "filter failed" and #applied == nA, { ok = okF, why = whyF, applied = #applied - nA })
step()
Text.submit(author, "first")
local okC, whyC = Text.submit(author, "second")
case("a second submit inside the cooldown is refused", okC == false and whyC == "too fast", { ok = okC, why = whyC })

Text.forget(author)
Text.service, Text.now, Text.apply = real.service, real.now, real.apply
local passed = 0
for _, c in ipairs(cases) do
	if c.pass then
		passed += 1
	end
end
local j = HttpService:JSONEncode({ template = "filtered_text", name = "__NAME__", passed = passed, total = #cases, cases = cases })
--@emit MIFSELFTEST __NAME__.filtered_text
'''


# every source that prints to the MCP side takes the one emitter (see EMIT_FN above)
for _src in ("RUNNER", "FUZZ_SERVER", "SAVE_SELFTEST", "ACTION_SELFTEST", "PROMPT_SELFTEST", "ECONOMY_SELFTEST",
             "SCENERY_SELFTEST", "WARP_SELFTEST", "SUB_SELFTEST", "ANALYTICS_SELFTEST", "FILTER_SELFTEST"):
    globals()[_src] = with_emitter(globals()[_src])
del _src


def render(source, name, **tokens):
    """A template's source for one name: __NAME__ and the shared seam; nothing else is substituted."""
    out = source.replace("__SEAM__", SEAM.rstrip("\n")).replace("__NAME__", name)
    for k, v in tokens.items():
        out = out.replace("__%s__" % k.upper(), v)
    return out
