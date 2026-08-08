#include "HarnessTargetActor.h"

#include "Components/StaticMeshComponent.h"
#include "Components/TextRenderComponent.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "UObject/ConstructorHelpers.h"

AHarnessTargetActor::AHarnessTargetActor()
{
    PrimaryActorTick.bCanEverTick = false;
    Mesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("TargetMesh"));
    RootComponent = Mesh;
    static ConstructorHelpers::FObjectFinder<UStaticMesh> Cube(TEXT("/Engine/BasicShapes/Cube.Cube"));
    if (Cube.Succeeded())
    {
        Mesh->SetStaticMesh(Cube.Object);
    }
    Mesh->SetWorldScale3D(FVector(2.0f));

    Label = CreateDefaultSubobject<UTextRenderComponent>(TEXT("TargetLabel"));
    Label->SetupAttachment(Mesh);
    Label->SetRelativeLocation(FVector(0, 0, 80));
    Label->SetHorizontalAlignment(EHTA_Center);
    Label->SetWorldSize(32.0f);
}

void AHarnessTargetActor::OnConstruction(const FTransform& Transform)
{
    Super::OnConstruction(Transform);
    Label->SetText(FText::FromString(TargetLabel));
    Label->SetTextRenderColor(TargetColor.ToFColor(true));
    if (UMaterialInstanceDynamic* Material = Mesh->CreateAndSetMaterialInstanceDynamic(0))
    {
        Material->SetVectorParameterValue(TEXT("Color"), TargetColor);
    }
    Tags.AddUnique(TEXT("HarnessFixture"));
}

