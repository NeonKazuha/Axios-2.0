# Axios 2.0

Welcome to **Axios 2.0**! This repository hosts a transformer-based language model trained from scratch using PyTorch. The model is built on a dataset containing all of Shakespeare's works, designed to generate coherent and stylistically rich text.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

**Axios 2.0** is an upgraded version of the original Axios project. The goal is to create a robust generative model that captures the intricate language and stylistic elements present in Shakespeare's works. This repository not only includes the model training and evaluation scripts but also a user-friendly frontend built with Streamlit.

## Features

- **Transformer Architecture**: Built with PyTorch, focusing on generating high-quality text.
- **Shakespeare Dataset**: Trained on a dataset containing all of Shakespeare's works, ensuring the model produces text that is rich in classical language and poetic structure.
- **Streamlit Frontend**: Provides a seamless interface for users to interact with the model.
- **Modular Design**: Easy to extend and modify, with clear separation of concerns in the codebase.
- **MIT License**: Open-source and free to use.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/NeonKazuha/Axios-2.0.git
   cd Axios-2.0

2. Install the required dependencies:

   pip install -r requirements.txt

## Usage

To generate text using the trained model:

1. Launch the Streamlit frontend:

   ```bash
   streamlit run app.py

2. Access the application in your web browser and start generating text inspired by Shakespearean language.

## Dataset

The dataset used for training consists of all of Shakespeare's works. The dataset is preprocessed to fit the input requirements of the transformer model, ensuring that the generated text retains the stylistic elements characteristic of Shakespeare.

## Results

The model generates text that is thematically consistent with Shakespeare's writing. While not perfect, the results are promising and demonstrate the model's ability to produce text that is rich in classical language and poetic structure.

## Future Work

- **Further Fine-Tuning**: Experimenting with additional datasets and fine-tuning techniques to improve output quality.
- **Incorporating Attention Mechanisms**: Investigating the use of different attention mechanisms for better performance.
- **Expanding the Dataset**: Including other classical literary works to diversify the training data.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

Special thanks to the open-source community and contributors who made this project possible.
