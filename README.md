# Bites and Bytes: NLP for Italian Gastronomy

## Project Overview
This repository hosts the code and data for the publication <em>On Cross-Language Entity Label Projection and Recognition</em>, a study focused on applying Natural Language Processing techniques to the culinary field. Conducted under the auspices of the University of Bologna's PhD program in Translation, Interpreting, and Interculturality, this research is part of a scholarship funded by PNRR (National Recovery and Resilience Plan) ex D.M. 118/2023. The project aims to promote the Italian culinary heritage using a multilingual approach.

## Research Objectives and Methodology
The primary objective is to develop a computational system capable of adapting omnivorous recipes to dietary restrictions such as vegetarian, vegan, halal, or kosher diets. The current pipeline, partially implemented and illustrated below, enriches culinary data by annotating various entities like "food", "quantity", and "process".

![Pipeline](https://i.imgur.com/zX51MP1.png "Project pipeline")
*Figure: Recipe adaptation pipeline. The green section is implemented; the red section is planned.*

This enriched data is used to train models to extract such entities from recipes, which are crucial for describing and modeling a culinary recipe. These structured data are then used to develop a model that can substitute unwanted ingredients in a recipe. The adapted recipe is provided to a large language model, which outputs a complete recipe with ingredients and preparation instructions.

## Repository Structure
- **data/**: Contains the datasets and annotations used in the project, including multilingual adaptations.
- **src/**: Source code for the project.
  - **align/**: Scripts for data alignment and translation.
  - **extract/**: Data extraction and preprocessing scripts.
  - **ner/**: Named entity recognition models and utilities.
  - **recipe_classifier/**: Scripts for training and testing recipe classification models.
  - **translate/**: Utilities for translating and localizing recipes.
- **requirements.txt**: Dependencies required for the project.

## Installation
To set up the environment for the project, follow these steps:
```
git clone https://github.com/paolo-gajo/food.git
cd food
pip install -r requirements.txt
```

## Usage
Details on how to use the scripts and models for data preprocessing, training, and recipe adaptation are provided in the respective directories within `src/`.

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your suggested changes.

## License
This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Acknowledgments
This project is funded by the National Recovery and Resilience Plan under Grant Agreement No. 118/2023.
