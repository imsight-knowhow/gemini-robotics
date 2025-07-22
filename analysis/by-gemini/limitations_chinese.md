# Gemini Robotics 模型局限性总结 (中文)

根据源论文第6节“讨论”以及其他章节的内容，Gemini Robotics系列模型存在以下主要局限性和未来工作的方向：

-   **长时间视频中的空间关系理解能力不足 (Difficulty with Long Videos)**
    -   **描述:** 模型在处理和理解跨越较长时间段的视频时，可能难以准确地把握物体和环境的空间关系。
    -   **论文引用:**
        > "For example, Gemini 2.0 may struggle with grounding spatial relationships across long videos..." (Section 6)

-   **预测精度问题 (Insufficient Prediction Precision)**
    -   **描述:** 模型生成的数值预测（例如，2D/3D边界框和定位点）的精度可能不足以支持更精细、高要求的机器人控制任务。
    -   **论文中的例子:** 在零样本（Zero-shot）控制实验中，`Gemini Robotics-ER` 模型无法完成折叠裙子的任务，主要原因就是其生成的抓取点不够精确。
    > "For a harder and more dexterous task: Gemini Robotics-ER is currently unable to perform dress folding, mostly due to its inability to generate precise enough grasps." (Section 2.3)
    > "...and its numerical predictions (e.g., points and boxes) may not be precise enough for more fine-grained robot control tasks." (Section 6)

-   **复杂场景下的多步推理与精确执行结合能力有待提升 (Weakness in Combining Multi-Step Reasoning with Dexterity in Novel Scenarios)**
    -   **描述:** 在全新的、需要多步骤抽象推理和高精度灵巧操作相结合的复杂场景中，模型的表现仍有提升空间。
    -   **论文中的例子:** 在多任务学习设置中，一些最灵巧的任务（如“穿鞋带”）对通用模型来说仍然非常具有挑战性。
    > "...we find that some of the most dexterous tasks are still quite challenging to learn purely from the multi-task setup (e.g., “insert shoe lace”)..." (Section 3.2)
    > "First, we aim to enhance Gemini Robotics’s ability to handle complex scenarios requiring both multi-step reasoning and precise dexterous movements, particularly in novel situations." (Section 6)

-   **对新机器人形态的适应性仍需大量数据 (Data-Intensive Adaptation for New Embodiments)**
    -   **描述:** 虽然模型可以适应新的机器人形态（Embodiments），但这个过程仍然需要一定量的目标平台数据。最终目标是实现零样本的跨形态迁移，但这尚未实现。
    -   **论文引用:**
        > "Finally, we will expand our multi-embodiment experiments, aiming to reduce the data needed to adapt to new robot types and ultimately achieve zero-shot cross-embodiment transfer..." (Section 6)

-   **对模拟数据的依赖和Sim-to-Real的挑战 (Reliance on Simulation and Sim-to-Real Challenges)**
    -   **描述:** 未来的发展计划更多地依赖模拟环境来生成多样化和富含接触（contact-rich）的数据，但这本身也带来了如何将模拟环境中学习到的能力有效迁移到现实世界（Sim-to-Real）的挑战。
    -   **论文引用:**
        > "Second, we plan to lean more on simulation to generate visually diverse and contact rich data as well as developing techniques for using this data to build more capable VLA models that can transfer to the real world..." (Section 6)
