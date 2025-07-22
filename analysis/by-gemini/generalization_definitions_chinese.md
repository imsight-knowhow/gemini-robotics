# 泛化能力定义 (Simplified Chinese)

根据原始论文，对三种关键泛化能力的定义翻译如下：

### 1. 视觉泛化能力 (Visual Generalization)
模型应能应对场景中不影响完成任务所需动作的视觉变化。这些视觉变化可以包括背景、光照条件、干扰物体或纹理的变化。

**原文定义 (Original Definition):**
> Visual Generalization: The model should be invariant to visual changes of the scene that do not affect the actions required to solve the task. These visual changes can include variations in background, lighting conditions, distractor objects or textures.

---

### 2. 指令泛化能力 (Instruction Generalization)
模型应能理解自然语言指令中的不变性和等效性。除了第3.3节中研究的精细可控性之外，模型还应能理解转述、对拼写错误具有鲁棒性、理解不同语言以及不同程度的指令特异性。

**原文定义 (Original Definition):**
> Instruction Generalization: The model should understand invariance and equivalence in natural language instructions. Going beyond fine-grained steerability studied in Section 3.3, the model should understand paraphrasing, be robust to typos, understand different languages, and varying levels of specificities.

---

### 3. 动作泛化能力 (Action Generalization)
模型应能调整已学会的动作或合成新动作，以适应例如训练期间未见过的初始条件（如物体放置位置）或物体实例（如不同的形状或物理属性）的泛化。

**原文定义 (Original Definition):**
> Action Generalization: The model should be capable of adapting learned movements or synthesizing new ones, for instance to generalize to initial conditions (e.g., object placement) or object instances (e.g., shape or physical properties) not seen during training.

---

## 核心概念要点 (Core Concepts Summary)

-   **视觉泛化 (Visual Generalization):**
    -   **目标:** 对与任务无关的视觉变化保持不变性。
    -   **具体变化:**
        -   背景 (Background)
        -   光照 (Lighting)
        -   干扰物 (Distractor Objects)
        -   纹理 (Textures)

-   **指令泛化 (Instruction Generalization):**
    -   **目标:** 理解自然语言指令的内在含义，不受表达方式影响。
    -   **具体能力:**
        -   理解意译和转述 (Paraphrasing)
        -   抵抗拼写错误 (Robustness to Typos)
        -   跨语言理解 (Different Languages)
        -   适应不同详细程度的指令 (Varying Levels of Specificity)

-   **动作泛化 (Action Generalization):**
    -   **目标:** 调整或创造动作以适应新情况。
    -   **具体场景:**
        -   新的初始条件 (New Initial Conditions)，例如物体位置不同。
        -   新的物体实例 (New Object Instances)，例如不同的物体形状或物理特性。
