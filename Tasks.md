# Tasks     

- ### Automation Dataset Updates
  - Create a script that clones or updates the repo using Git commands (git pull for updates)
  - Automate downloading and processing the latest dataset files
  - Include the script in the data pipeline, triggering it before the transformation step to ensure updated data
  - After fetching the data, validate the new dataset's structure and integrate it into the pipeline

- ### HANDLE DATA
  **Creating a Data Transformation Pipeline**
    - ~~Save the CSV files~~
    - ~~Data Cleaning: Handle missing values, standardize formats, and filter irrelevant records~~
    - ~~Aggregation: Create the desired metrics, summaries, or formats (e.g., player stats per season)~~
    - ~~Output: Save the transformed data into a new directory (e.g., /processed/)~~

- ### Define Team Selection Rules
  - **Create a ruleset for a valid team. Examples:**
    - maximum budget constraint
    - minimum and maximum players from specific positions.
    - restrictions on players from the same team.
    - overall team size constraint.

- ### Implement the basic algorithm for selecting a valid team:
  - Input: Budget, number of players, and other constraints.
  - Output: A team that satisfies all rules.

- ### Build the FF-Agent Interface
  - input: list of matches, date, round, budget
  - output the generated team.
  - add performance metrics (Optional)

- ### Enhance the agent with simple metrics like:
  - Average historical performance of selected players.
  - Expected points per cost.
  - Testing and Validation

- ### Test with multiple inputs:
  - Teams are always valid.
  - Rules and constraints are properly enforced.
  - Compare outputs against edge cases (e.g., lowest possible budget, maximum players from a team).
  - Iteration and Refinement

- ### Documentation and Deployment
  - Document how to use the FF-Agent.
  - Deploy the agent for easy access (e.g., CLI, API, or lightweight web app).
- 
- ### FUTURE WORK
  - LangChain
  - RAG methods
  - add Reinforcement Learning logic (punish/reward, metric to estimate how good is a team, etc...) 
