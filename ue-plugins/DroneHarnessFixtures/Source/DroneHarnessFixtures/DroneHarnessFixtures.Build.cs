using UnrealBuildTool;

public class DroneHarnessFixtures : ModuleRules
{
    public DroneHarnessFixtures(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine" });
    }
}

