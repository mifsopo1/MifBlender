"""the internal notes's domain classifier, one home for it: a class goes to the first rule whose name pattern matches
it, or whose named base is in its superclass chain; anything else is "instances and parts". Used by
a dev tool (the citation and denominator check), a dev tool (a new build's
changes by domain) and rbx_coverage (coverage per domain), so the three can never split a class differently. Moved
here from roblox_catalogue_check.py, which does not ship with the server.
"""
import collections
import re

from mif_roblox import catalog as C

# the internal notes, in order: (domain, class-name pattern, superclass names or None).
RULES = [
    ("packages", r"^(PackageLink|PackageService|PackageUIService|AssetPatchSettings|IncrementalPatchBuilder|PatchMapping|PatchBundlerFileWatch)$", None),
    ("collaboration", r"(TeamCreate|Collab|LiveSync|InstanceFileSync|SyncScriptBuilder|Draft|VersionControl|ScriptChange|CommentService|AnnotationService|Annotation$|MultiplayerEmulation)", None),
    ("avatar and UGC", r"(^Avatar|HumanoidDescription|HumanoidRigDescription|DigitsRigDescription|AccessoryDescription|BodyPartDescription|MakeupDescription|^Accessory|Accoutrement|^Clothing$|^Shirt|^Pants$|Wrap(Layer|Target|Deformer|TextureTransfer)|^BaseWrap$|^BodyColors$|^CharacterMesh$|^FaceControls$|^Skin$|^Makeup|^Hat$|UGC|^AvatarEffect)", None),
    ("animation and rigs", r"(^Animat|Keyframe|^Pose|Curve|^Bone$|IKControl|BoneBehavior|RTAnimation|TrackerLod|StateMachine|^Humanoid$|^Motor6D$|FaceAnimator|^Animator$|ClipEvaluator)", None),
    ("audio", r"(^Audio|^Sound|SoundEffect$|VoiceChat|^RealtimeMedia$|ParticipantHandle|^SoundGroup$)", "SoundEffect"),
    ("terrain and materials", r"(^Terrain|^MaterialService$|^MaterialVariant$|^SurfaceAppearance$|^TerrainDetail$|^TerrainRegion$)", None),
    ("lighting and effects", r"(^Lighting$|^Sky$|^Atmosphere$|^Clouds$|^Highlight$|^RenderSettings$|^ParticleEmitter|^Beam$|^Trail$|^Fire$|^Smoke$|^Sparkles$|^Explosion$|^ForceField$|^FaceInstance$|^Decal$|^Texture$)", "Light|PostEffect"),
    ("editor, selection and plugin", r"(^Studio$|^Plugin|^Selection$|^ChangeHistoryService$|^DraggerService$|^StudioService$|^StudioTheme$|^DockWidgetPluginGui$|^PluginGui$|^Dragger$|^ReflectionMetadata|^ExplorerServiceVisibility|^StudioAttachment$|^UniqueIdLookupService$|^Handles$|^ArcHandles$|^HandlesBase$|^IntegrityService$|^PlaceSettings$|^ScriptProfiler)", None),
    ("UI", r"(^Gui|^UI|^Style|^StarterGui$|^PlayerGui$|^BasePlayerGui$|^CoreGui$|^TextService$|^GetTextBoundsParams$|^Localization|^Translator$|^LanguageService$|CoreGuiConfiguration|Configuration$|^ScreenshotHud$|^Path2D$)", "GuiBase|UIComponent|StyleBase"),
    ("physics", r"(^Physics|^Pathfinding|^Path$|Controller|Sensor$|^SensorBase$|^NoCollisionConstraint$|^WorldRoot$|^WorldModel$|^ControlState$|^Glue$|^Motor$|^VelocityMotor$|^RocketPropulsion$|^Skateboard|^FluidForceSensor$|^NoCollision)", "Constraint|JointInstance|BodyMover|ControllerBase|SensorBase"),
    ("camera and input", r"(^Camera$|InputService$|^ContextActionService$|^Input(Action|Binding|Context|Object)|^InputActionLabel$|^Mouse$|^PluginMouse$|^ClickDetector$|^DragDetector$|^UIDragDetector$|^ProximityPrompt|^Haptic|^VRService$|^GamepadService$|^UserGameSettings$|^VirtualInput$|^UserSettings$|^GlobalSettings$)", None),
    ("play-test and diagnostics", r"(^Test|Test$|TestService$|^Stats|^LogService$|Profiler|^DebugSettings$|Debugger|^RunService$|^RenderingTest$|Capture|^ScreenshotCapture|^Studio(Test|DeviceSimulator|ScreenshotCapture|CaptureService)|^Network|^Logger$|^TaskScheduler$|^ScriptContext$|^SceneAnalysisService$|^MemStorage|^FunctionalTest$|^TestCase$|^CustomLog$|^CustomEvent|RemoteCommand|^ExecutedRemoteCommand$|^MLService$|^MLSession$)", None),
    ("scripts", r"(^ScriptEditorService$|^ScriptDocument$|^ScriptService$|^Remote(Event|Function)$|^UnreliableRemoteEvent$|^Bindable|^Actor$|^SharedTableRegistry$|^AuroraScript|^ReflectionService$)", "LuaSourceContainer"),
    ("data", r"(DataStore|MemoryStore|^HttpService$|^HttpRbxApiService$|^MessagingService$|^Config(Service|Snapshot)$|^SerializationService$|^EncodingService$|^WebSocket|^WebStream|^QueueService$|^StandardQueue$)", None),
    ("assets and import", r"(^AssetService$|^InsertService$|^ContentProvider$|ImportData$|^ImportSession$|^Editable(Mesh|Image)$|^GenerationService$|^FileMesh$|^SpecialMesh$|^DataModelMesh$|^AssetDelivery|^RuntimeContentService$|^File$|^ProceduralModel$|^GeneratedFolder$|^VideoService$|^Video(Player|Sampler|DeviceInput|Capture|CaptureService)$|^FeatureMatching|^FeatureSearchParams$|^Feature$)", None),
    ("gameplay and social services", r"(^Players?$|^Teams?$|Marketplace|Badge|^Ad(Service|Portal|Gui|Placement|Reward)$|^Social|Chat|^TeleportService$|^Teleport|^GroupService$|^PolicyService$|^PointsService$|Commerce|Recommendation|Analytics|Moderation|Notification|Matchmaking|^StarterPlayer|^StarterGear$|^Tool$|^HopperBin$|^BackpackItem$|^Backpack$|^SpawnLocation$|^Dialog|^Flag|^ExperienceInvite|^UserService$|^PlayerScripts$|^PlayerViewService$|^TextChannel|^TextChatCommand$|^TextFilter|^AppHandle$|^Romark)", None),
]
OTHER = "instances and parts"


def domain(cat, cls):
    chain = cat.chain(cls)
    for name, rx, base in RULES:
        if re.search(rx, cls):
            return name
        if base and any(re.fullmatch(base, b) for b in chain):
            return name
    return OTHER


def domains(cat):
    """{domain: {"classes": n, "members": n, "classNames": [...]}} over counted members."""
    out = collections.defaultdict(lambda: {"classes": 0, "members": 0, "classNames": []})
    for cls, c in cat.classes.items():
        n = sum(1 for row in c["m"].values() if C.is_counted(row))
        if n:
            d = out[domain(cat, cls)]
            d["classes"] += 1
            d["members"] += n
            d["classNames"].append("%s:%d" % (cls, n))
    return dict(out)
