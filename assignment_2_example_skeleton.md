# Assignment 2: MLOps & PCAM Pipeline Journal
**MLOps & ML Programming (2026)**

## Group Information
* **Group Number:** [25]
* **Team Members:** [Member 1 Lawrence 15277844, Member 2 Michael Dong 15804232, Member 3 Dennis Chan 15833526, Member 4 Ngawang Tsarong 15833526]
* **GitHub Repository:** [https://github.com/Denferno/MLOps_2026]
* **Base Setup Chosen from Assignment 1:** [Dennis Chan]

---

## Question 1: Reproducibility Audit
1. **Sources of Non-Determinism:**

   1. Pytorch CUDA determinism (GPU)
Wanneer je GPU (CUDA) gebruikt, dan gebruikt PyTorch niet deterministische algoritmes voor de snelheid. Zelfs als we alle seeds erbij zetten kunnen sommige berekeningen toch net anders zijn waardoor het net een andere resultaat produceert. 

   2. DataLoader "Shuffle" Probleem.
   Dataloader die shuffelt (door shuffle=True) de data waardoor alle waardes een willekeurige waarde krijgen en dat zorgt ervoor dat het niet meer deterministisch is.

   3. Verschillende CPUs en GPu maken de exact dezelfde berekingen maar net iets anders, omdat computers geen perfecte decimalen kunnen opslaan en ronden tussendoor ook af en daardoor kunnen later er verschillen ontstaan in de training en hierdoor is perfect identiek niet haalbaar.  

   4. 


2. **Control Measures:**

3. **Code Snippets for Reproducibility:**
   ```python
   # Paste the exact code added for seeding and determinism
   ```

4. **Twin Run Results:**

---

## Question 2: Data, Partitioning, and Leakage Audit
1. **Partitioning Strategy:**

2. **Leakage Prevention:**
   
3. **Cross-Validation Reflection:**

4. **The Dataset Size Mystery:**

5. **Poisoning Analysis:**

---

## Question 3: Configuration Management
1. **Centralized Parameters:**

2. **Loading Mechanism:**
   - [Describe your use of YAML, Hydra, or Argparse.]
   ```python
   # Snippet showing how parameters are loaded
   ```

3. **Impact Analysis:**

4. **Remaining Risks:** 

---

## Question 4: Gradients & LR Scheduler
1. **Internal Dynamics:**

2. **Learning Rate Scheduling:**

---

## Question 5: Part 1 - Experiment Tracking
1. **Metrics Choice:**

2. **Results (Average of 3 Seeds):**

3. **Logging Scalability:**

4. **Tracker Initialization:**
   ```python
   # Snippet showing tracker/MLFlow/W&B initialization
   ```

5. **Evidence of Logging:**

6. **Reproduction & Checkpoint Usage:**

7. **Deployment Issues:**

---

## Question 5: Part 2 - Hyperparameter Optimization
1. **Search Space:**
2. **Visualization:**
3. **The "Champion" Model:**

4. **Thresholding Logic:**

5. **Baseline Comparison:**

---

## Question 6: Model Slicing & Error Analysis
1. **Visual Error Patterns:**

2. **The "Slice":**

3. **Risks of Silent Failure:**

---

## Question 7: Team Collaboration and CI/CD
1. **Consolidation Strategy:** 

We hebben de SURF MLops github als foundation gekozen omdat die natuurlijk compleet was, maar vooral omdat git merge veel conflicten gaf. Eerst wilde ik git merge doen, want dan zou ik ook nog paar van mijn eigen files bewaren, maar het ging niet helemaal goed. (zie question 7.4)**


2. **Collaborative Flow:**

3. **CI Audit:**

4. **Merge Conflict Resolution:**
Ik probeerde eerst `git remote add surf` en vervolgens `git fetch`. Toen had ik veel merge conflicts. Ik deed in mijn terminal `git status` en zag ik veel merge conflicts had. Ik probeerde het op te lossen de terminal eerst met `git mergetool`, maar ik wist niet precies hoe dat werkte. Daarom had ik het opgelost in visual studio code, maar ik had geen idee of ik het goed had gedaan. Ik had namelijk alle incoming geaccepteerd. Toen wilde ik checken of alle files van https://github.com/SURF-ML/MLOps_2026 overeenkwam met onze repository. Dus ik dacht dan doe ik opnieuw git fetch surf, maar het was "already up to date" . Toen probeerde ik git push origin main --force en nu is mijn eigen werk van assigment 1 weg. 

5. **Branching Discipline:**

---

## Question 8: Benchmarking Infrastructure
1. **Throughput Logic:**

2. **Throughput Table (Batch Size 1):**

| Partition | Node Type | Throughput (img/s) | Job ID |
| :--- | :--- | :--- | :--- |
| `thin_course` | CPU Only | | |
| `gpu_course` | GPU ([Type]) | | |

3. **Scaling Analysis:**

4. **Bottleneck Identification:**

---

## Question 9: Documentation & README
1. **README Link:** [Link to your Group Repo README]
2. **README Sections:** [Confirm Installation, Data Setup, Training, and Inference are present.]
3. **Offline Handover:** [List the files required on a USB stick to run the model offline.]

---

## Final Submission Checklist
- [ ] Group repository link provided?
- [ ] Best model checkpoint pushed to GitHub?
- [ ] inference.py script included and functional?
- [ ] All Slurm scripts included in the repository?
- [ ] All images use relative paths (assets/)?
- [ ] Names and IDs of all members on the first page?