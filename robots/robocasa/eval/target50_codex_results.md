# Codex Target50 Results

This page reports the task-level aggregate results for the 340-cell RoboCasa
Target50 evaluation defined by [`target50.json`](target50.json). The reference
planner profile is Codex SDK with `gpt-5.5`, `xhigh` reasoning effort, and
`max_turns=100`. Success is determined only by the environment's
`state.success` value.

The published record contains aggregate success counts for each task. It does
not include per-seed traces, raw trajectories, failure classifications, or
planner credentials, so it supports task-level comparison rather than
independent per-cell auditing.

## Split Summary

| Split | Tasks | Reproduction | Success rate | Harness VLA reference | Difference |
|---|---:|---:|---:|---:|---:|
| Atomic | 18 | 163/180 | 90.56% | 165/180 (91.67%) | -1.11 pp |
| Composite-Seen | 16 | 49/80 | 61.25% | 45/80 (56.25%) | +5.00 pp |
| Composite-Unseen | 16 | 12/80 | 15.00% | 11/80 (13.75%) | +1.25 pp |
| **Overall (task-weighted)** | **50** | **N/A** | **57.00%** | **55.40%** | **+1.60 pp** |

Split rates use `successful cells / evaluated cells`. The overall result uses
the task-weighted RoboCasa365 convention, under which every task contributes
equally despite Atomic having ten seeds per task and the composite splits
having five:

```text
(18 * 90.555556% + 16 * 61.25% + 16 * 15.00%) / 50 = 57.00%
```

The reference values are the corresponding Harness VLA Codex counts reported
for RoboCasa365. See the [Harness VLA paper](https://arxiv.org/abs/2607.08448)
and the [RPent overview](../../../docs/source-en/rst_source/awesome_works/harnessvla.rst).

## Per-Task Results

| # | Split | Task | Successful cells | Success rate |
|---:|---|---|---:|---:|
| 1 | Atomic | CloseBlenderLid | 5/10 | 50% |
| 2 | Atomic | CloseFridge | 8/10 | 80% |
| 3 | Atomic | CloseToasterOvenDoor | 9/10 | 90% |
| 4 | Atomic | CoffeeSetupMug | 9/10 | 90% |
| 5 | Atomic | NavigateKitchen | 9/10 | 90% |
| 6 | Atomic | OpenCabinet | 10/10 | 100% |
| 7 | Atomic | OpenDrawer | 9/10 | 90% |
| 8 | Atomic | OpenStandMixerHead | 10/10 | 100% |
| 9 | Atomic | PickPlaceCounterToCabinet | 9/10 | 90% |
| 10 | Atomic | PickPlaceCounterToStove | 10/10 | 100% |
| 11 | Atomic | PickPlaceDrawerToCounter | 9/10 | 90% |
| 12 | Atomic | PickPlaceSinkToCounter | 10/10 | 100% |
| 13 | Atomic | PickPlaceToasterToCounter | 8/10 | 80% |
| 14 | Atomic | SlideDishwasherRack | 9/10 | 90% |
| 15 | Atomic | TurnOffStove | 10/10 | 100% |
| 16 | Atomic | TurnOnElectricKettle | 10/10 | 100% |
| 17 | Atomic | TurnOnMicrowave | 9/10 | 90% |
| 18 | Atomic | TurnOnSinkFaucet | 10/10 | 100% |
| 19 | Composite-Seen | ScrubCuttingBoard | 5/5 | 100% |
| 20 | Composite-Seen | StackBowlsCabinet | 5/5 | 100% |
| 21 | Composite-Seen | WashLettuce | 4/5 | 80% |
| 22 | Composite-Seen | RinseSinkBasin | 5/5 | 100% |
| 23 | Composite-Seen | PreSoakPan | 2/5 | 40% |
| 24 | Composite-Seen | StirVegetables | 2/5 | 40% |
| 25 | Composite-Seen | LoadDishwasher | 2/5 | 40% |
| 26 | Composite-Seen | SteamInMicrowave | 3/5 | 60% |
| 27 | Composite-Seen | SetUpCuttingStation | 1/5 | 20% |
| 28 | Composite-Seen | GetToastedBread | 1/5 | 20% |
| 29 | Composite-Seen | DeliverStraw | 0/5 | 0% |
| 30 | Composite-Seen | KettleBoiling | 3/5 | 60% |
| 31 | Composite-Seen | PrepareCoffee | 5/5 | 100% |
| 32 | Composite-Seen | StoreLeftoversInBowl | 4/5 | 80% |
| 33 | Composite-Seen | SearingMeat | 4/5 | 80% |
| 34 | Composite-Seen | PackIdenticalLunches | 3/5 | 60% |
| 35 | Composite-Unseen | ArrangeBreadBasket | 1/5 | 20% |
| 36 | Composite-Unseen | ArrangeTea | 1/5 | 20% |
| 37 | Composite-Unseen | BreadSelection | 1/5 | 20% |
| 38 | Composite-Unseen | CategorizeCondiments | 2/5 | 40% |
| 39 | Composite-Unseen | CuttingToolSelection | 1/5 | 20% |
| 40 | Composite-Unseen | GarnishPancake | 0/5 | 0% |
| 41 | Composite-Unseen | GatherTableware | 1/5 | 20% |
| 42 | Composite-Unseen | HeatKebabSandwich | 0/5 | 0% |
| 43 | Composite-Unseen | MakeIceLemonade | 1/5 | 20% |
| 44 | Composite-Unseen | PanTransfer | 0/5 | 0% |
| 45 | Composite-Unseen | PortionHotDogs | 1/5 | 20% |
| 46 | Composite-Unseen | RecycleBottlesByType | 1/5 | 20% |
| 47 | Composite-Unseen | SeparateFreezerRack | 0/5 | 0% |
| 48 | Composite-Unseen | WaffleReheat | 1/5 | 20% |
| 49 | Composite-Unseen | WashFruitColander | 1/5 | 20% |
| 50 | Composite-Unseen | WeighIngredients | 0/5 | 0% |
