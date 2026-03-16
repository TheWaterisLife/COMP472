# Project Timeline: ASL Recognition Using CNNs

**Course:** COMP 472: Applied Artificial Intelligence  
**Term:** Winter 2026

---

## Gantt Chart (PlantUML)

```plantuml
@startgantt
title ASL Recognition Project Timeline

printscale weekly
Project starts 2026-01-12

-- Phase 1: Preparation --
[Literature Review] starts 2026-01-12 and lasts 2 weeks
[Dataset Acquisition & Exploration] starts 2026-01-12 and lasts 2 weeks
[Environment Setup (PyTorch)] starts 2026-01-19 and lasts 2 weeks
[Milestone 1: Data Ready] happens at [Environment Setup (PyTorch)]'s end

-- Phase 2: Model Development --
[Data Preprocessing Pipeline] starts 2026-01-26 and lasts 2 weeks
[Baseline CNN Implementation] starts 2026-02-02 and lasts 2 weeks
[ResNet Implementation] starts 2026-02-09 and lasts 2 weeks
[MobileNet Implementation] starts 2026-02-16 and lasts 2 weeks
[Hyperparameter Tuning] starts 2026-02-23 and lasts 2 weeks
[Milestone 2: Models Trained] happens at [Hyperparameter Tuning]'s end

-- Phase 3: Evaluation & Analysis --
[Performance Evaluation] starts 2026-03-02 and lasts 2 weeks
[Comparative Analysis] starts 2026-03-09 and lasts 2 weeks
[Error Analysis & Visualization] starts 2026-03-09 and lasts 2 weeks
[Milestone 3: Results Complete] happens at [Comparative Analysis]'s end

-- Phase 4: Documentation --
[Final Report Writing] starts 2026-03-16 and lasts 2 weeks
[Code Documentation] starts 2026-03-23 and lasts 2 weeks
[Presentation Preparation] starts 2026-03-23 and lasts 2 weeks
[Final Submission] happens at [Presentation Preparation]'s end

@endgantt
```

---

## Alternative: Simplified PlantUML Gantt

```plantuml
@startgantt
title COMP 472 - ASL Recognition Project

printscale weekly
saturday are closed
sunday are closed

Project starts 2026-01-12

[Literature Review] lasts 14 days
[Dataset Acquisition] lasts 14 days
[Environment Setup] lasts 14 days
then [Data Preprocessing] lasts 14 days
then [Baseline CNN] lasts 14 days
then [ResNet Model] lasts 14 days
then [MobileNet Model] lasts 14 days
then [Hyperparameter Tuning] lasts 14 days
then [Evaluation] lasts 14 days
then [Report Writing] lasts 14 days
then [Final Submission] lasts 7 days

[Dataset Acquisition] starts at [Literature Review]'s start
[Environment Setup] starts at [Literature Review]'s end

@endgantt
```

---

## Key Milestones

| Milestone | Week | Deliverables |
|-----------|------|--------------|
| **M1: Data Ready** | 3 | All three datasets downloaded, explored, and preprocessed; development environment configured |
| **M2: Models Trained** | 9 | Baseline CNN, ResNet, and MobileNet trained on all three datasets; hyperparameters optimized |
| **M3: Results Complete** | 12 | All evaluation metrics computed; comparative analysis across datasets and models completed |
| **Final Submission** | 14 | Final report, documented codebase, and presentation materials submitted |

---

## Task Breakdown

### Phase 1: Preparation (Weeks 1-3)
- [ ] Conduct literature review on ASL recognition and CNN architectures
- [ ] Download and explore all three datasets
- [ ] Set up PyTorch development environment
- [ ] Create data loading utilities

### Phase 2: Model Development (Weeks 4-9)
- [ ] Implement data preprocessing pipeline (resize, normalize, augment)
- [ ] Build and train baseline CNN architecture
- [ ] Implement and fine-tune ResNet-18/50 with transfer learning
- [ ] Implement and fine-tune MobileNetV2
- [ ] Perform hyperparameter optimization (learning rate, batch size, etc.)

### Phase 3: Evaluation & Analysis (Weeks 10-12)
- [ ] Evaluate all models on all three datasets
- [ ] Compute accuracy, F1-score, confusion matrices, ROC-AUC
- [ ] Perform comparative analysis across models and datasets
- [ ] Analyze common error patterns and failure cases
- [ ] Create visualizations (training curves, confusion matrices, sample predictions)

### Phase 4: Documentation (Weeks 12-14)
- [ ] Write final project report
- [ ] Document codebase with comments and README
- [ ] Prepare presentation slides
- [ ] Submit all deliverables

---

## How to Render PlantUML

You can render the Gantt chart using:

1. **PlantUML Online Server**: https://www.plantuml.com/plantuml/uml/
2. **VS Code Extension**: Install "PlantUML" extension
3. **Command Line**: `java -jar plantuml.jar Gantt_Chart.md`
4. **Jupyter Notebook**: Use `plantuml` magic with iplantuml package
