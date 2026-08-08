#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "HarnessTargetActor.generated.h"

class UStaticMeshComponent;
class UTextRenderComponent;

UCLASS()
class DRONEHARNESSFIXTURES_API AHarnessTargetActor : public AActor
{
    GENERATED_BODY()

public:
    AHarnessTargetActor();
    virtual void OnConstruction(const FTransform& Transform) override;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Harness")
    FLinearColor TargetColor = FLinearColor::Red;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Harness")
    FString TargetLabel = TEXT("HARNESS TARGET");

private:
    UPROPERTY()
    UStaticMeshComponent* Mesh;

    UPROPERTY()
    UTextRenderComponent* Label;
};

