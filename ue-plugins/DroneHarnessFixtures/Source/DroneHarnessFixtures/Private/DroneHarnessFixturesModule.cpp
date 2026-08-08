#include "HarnessTargetActor.h"

#include "Engine/World.h"
#include "HAL/IConsoleManager.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "Modules/ModuleManager.h"

class FDroneHarnessFixturesModule : public IModuleInterface
{
public:
    virtual void StartupModule() override
    {
        Handle = FWorldDelegates::OnPostWorldInitialization.AddRaw(
            this, &FDroneHarnessFixturesModule::OnWorldInitialized);
    }

    virtual void ShutdownModule() override
    {
        FWorldDelegates::OnPostWorldInitialization.Remove(Handle);
    }

private:
    void OnWorldInitialized(UWorld* World, const UWorld::InitializationValues)
    {
        if (!World || !World->IsGameWorld() || !FParse::Param(FCommandLine::Get(), TEXT("HarnessFixture")))
        {
            return;
        }
        float X = 1000.0f;
        float Y = 1000.0f;
        float Z = 100.0f;
        FParse::Value(FCommandLine::Get(), TEXT("HarnessTargetX="), X);
        FParse::Value(FCommandLine::Get(), TEXT("HarnessTargetY="), Y);
        FParse::Value(FCommandLine::Get(), TEXT("HarnessTargetZ="), Z);
        World->SpawnActor<AHarnessTargetActor>(FVector(X, Y, Z), FRotator::ZeroRotator);
    }

    FDelegateHandle Handle;
};

IMPLEMENT_MODULE(FDroneHarnessFixturesModule, DroneHarnessFixtures)

