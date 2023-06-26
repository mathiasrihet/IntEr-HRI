# IntEr-HRI

This repository contains all the code needed to reproduce the results of the ChocolaTeam during IntEr-HRI competition (IJCAI 2023).

```
📦IntEr-HRI
 ┣ 📂.git
 ┣ 📂Data    <-- EEG data from the competition are supposed to be downloaded here
 ┃ ┣ 📂EEG
 ┃ ┃ ┣ 📂test data
 ┃ ┃ ┣ 📂training data
 ┃ ┃ ┗ 📜readme EEG.txt
 ┃ ┗ 📜.gitkeep
 ┣ 📂Results
 ┣ 📂Scripts
 ┃ ┣ 📜main_pipeline.py
 ┃ ┣ 📜single_subject_prediction.py
 ┃ ┣ 📜single_subject_validation.py
 ┃ ┗ 📜windowing_raw.py
 ┣ 📜.gitignore
 ┗ 📜setup.py
 ```

Both training data (https://zenodo.org/record/7951044) and testing data (https://zenodo.org/record/7966275) are required in order to run whole pipeline with main_pipeline.py.
