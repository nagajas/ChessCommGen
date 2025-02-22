# Chess Commentary Generation using Natural Language Generation

## Course: DSL604 - Natural Language Processing

## Problem Statement
Generating natural language descriptions for chess moves is a complex task due to its dependency on both the game state and pragmatic context. Traditional approaches either lack flexibility or fail to provide diverse commentary styles, making automated commentary generation a challenging problem.

### Challenges in Chess Commentary Generation:
- Requires understanding of both game state and contextual dependencies.
- Existing datasets are either limited in scale or lack diverse commentary styles.

### Limitations of Previous Approaches:
- **Rule-based methods**: Rely on predefined strategies, limiting flexibility.
- **Neural models**: Constrained by small datasets, leading to less diverse and context-aware commentary.

### Our Enhanced Approach:
- Incorporates additional features like move history, advantages, threats, and opportunities.
- Aims to generate more engaging, human-like, and educational commentary for players of all skill levels.

## Dataset Collection
- **Tools Used**:
  - `requests` for fetching web pages.
  - `BeautifulSoup` for parsing and extracting relevant content.
- **Source Website**: [GameKnot](https://gameknot.com)
- **Dataset Statistics**:
  - Total dataset size: **11.6K games**
  - Move-commentary pairs extracted: **298K**

## Methodology
We implemented two models for generating chess commentary:

### Model 1
- Uses a transformer-based architecture.
- Incorporates game state, past move history, and piece positioning.
- Achieved **83% accuracy** with a **loss of 1.09**.
<img width="2133" alt="model1" src="https://github.com/user-attachments/assets/b970985f-6c30-487d-9ad8-cdd89af699ba" />


### Model 2
- Enhanced with additional features such as threat detection and positional advantages.
- Achieved **95% accuracy** with a **loss of 0.33**.
<img width="2411" alt="model2" src="https://github.com/user-attachments/assets/f9ac4a24-257c-488d-adaf-1b18b6775e45" />


## Results
| Model | Accuracy | Loss |
|--------|---------|------|
| Model 1 | 83% | 1.09 |
| Model 2 | 95% | 0.33 |

### Classifier Performance
- Overall Classifier Accuracy: **74.25%**

## Challenges Faced
- **Data Extraction**: Collecting diverse move-commentary pairs from online sources.
- **Data Cleaning**: Removing noise and irrelevant commentary.
- **Model Training**: Fine-tuning models to improve coherence and relevance of generated commentary.

## Example Output
![Example Chess Commentary](https://github.com/user-attachments/assets/befc85f7-1bbb-46dc-aa96-e2e86c278c52)

## Reference
- Vasudevan, D. et al., "Learning to Generate Chess Game Commentary from Real-World Data," in *Proceedings of ACL 2018*, [Link](https://aclanthology.org/P18-1154/).

