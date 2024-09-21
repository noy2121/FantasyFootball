# Fantasy Football Team Builder

## Project Overview

This project implements an AI-powered Fantasy Football Team Builder using advanced natural language processing techniques. It leverages Retrieval-Augmented Generation (RAG) and fine-tuned language models to create optimal fantasy football teams based on user input, historical player data, and current match information.

This project uses content from repo **transfermarkt-datasets** (https://github.com/dcaribou/transfermarkt-datasets) 
licensed under CC0 1.0 Universal.

## Features

- Retrieval-Augmented Generation (RAG) for contextual information retrieval
- Custom fine-tuned language model for team composition
- Flexible inference modes (fine-tuned and vanilla)
- Comprehensive evaluation metrics
- User-friendly command-line interface

## Installation

1. Clone the repository:
git clone https://github.com/your-username/fantasy-football-team-builder.git
cd fantasy-football-team-builder

2. Install dependencies:
`pip install -r requirements.txt`

3. Set up the data:
- Ensure you have the necessary CSV files in the `data/csvs` directory
- Run the RAG data preparation script:
`python main.py mode="build_rag"`

## Usage

The project supports multiple modes of operation, controlled via the `mode` parameter in the configuration file or command-line override.

### Fine-tuning

To fine-tune the model:
`python main.py mode="fine_tune"`

### Inference

To generate a fantasy team using the fine-tuned model:
`python main.py mode="inference"`

You will be prompted to enter match information in the following format:  
'''  
matches: [Team1 vs Team2, Team3 vs Team4, ...],  
round: Group Stage/Round of 16/Quarter-final/Semi-final/Final,  
season: YYYY/YY,  
date: YYYY-MM-DD,    
''''

### Evaluation

To evaluate the model's performance:
`python main.py mode=evaluate`

## Configuration

The project uses Hydra for configuration management. The main configuration file is located at `src/config/conf.yaml`. You can override configuration parameters via command-line arguments or by modifying the YAML file directly.

## Project Structure

- `src/`
  - `model/`: Contains the core model implementations
    - `fantasy_model.py`: Main FantasyModel class
    - `flash_attention.py`: FlashAttention wrapper for optimize performance
    - `fantasy_stats.py`: Player and club statistics processing
    - `rag/`: Contains all RAG related code
      - `fantasy_rag.py`: RAG dataset creation, and retrival
    - `trainer/`: Training processes and utils
      - `fantsy_data_collator.py`: Data Collator
      - `fantasy_dataset.py`: Load and preprocess datasets
      - `fantasy_trainer`: Model training, evaluation and loss calculations
      - `fantasy_loss.py`: Custom loss
    - `metrics/fantasy_metrics.py`: Custom metrics
  - `config/`: Configuration files
- `data/`: Data directory (data not include in repository)
- `main.py`: Entry point for all operations

## Future Work

- Implement LangChain for improved language model interactions
- Explore advanced RAG methods for better information retrieval
- Incorporate Reinforcement Learning for team optimization
- Utilize larger models for better performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
This README provides a high-level overview of your project, explains how to use it, and includes examples of running different modes. It's structured in a way that's familiar to programmers and follows common README conventions. You may want to adjust some details (like the repository URL) to match your specific project setup.



