# Logboek assignment 2
# Question 1
**Question 1.1 List at least four concrete sources of non-reproducibility in your current pipeline. Exam-
ples include (but are not limited to): random seeds, data loading, hardware, libraries, or
filesystem behavior.**



**Question 1.2 For each source, state:
• Whether it is currently controlled in your code
• How you would control it (or why you chose not to)**



**Question 1.3 Show the exact code snippet(s) you added or modified to improve reproducibility. Make
it clear which parts are added on-top or modified from the original code.**



**Question 1.4 Re-run the same training job twice with the same configuration from all your group mem-
ber’s logins.
• Did you obtain identical results?
• If not, which parts still differ and why?**



# Question 7 
**The Consolidation Strategy: How did your group choose which codebase to use as the
"foundation"? Describe the process of moving code from individual repos to the group
repo. Did you use git merge, git cherry-pick, git rebase or manual porting? Jus-
tify your choice.**

We hebben de SURF MLops github als foundation gekozen omdat die natuurlijk compleet was, maar vooral omdat git merge veel conflicten gaf. Eerst wilde ik git merge doen, want dan zou ik ook nog paar van mijn eigen files bewaren, maar het ging niet helemaal goed. (zie question 7.4)**

**Question 7.2 Collaborative Flow: Provide a link or a screenshot of a Pull Request (PR) in your group
repository that satisfies the following:
• It was opened from a feature branch into dev or main.
• It contains at least two review comments from different team members.
• It shows that the GitHub Actions CI checks passed before it was merged.**


**Question 7.3 CI Configuration Audit: Look at your .github/workflows/ci.yml.
• Why do we explicitly install torch with the –index-url
https://download.pytorch.org/whl/cpu flag in the CI environment? What
would happen to the GitHub runner if we used the standard GPU-enabled install?
• How does the CI ensure that a teammate doesn’t merge code that breaks your
PCAMDataset or MLP architecture?**


** Question 7.4 escribe the most complex Merge Conflict (if you had one, else skip this question, in any
case we will be able to check in the repo) your team encountered. Explain step-by-step
how you resolved it without losing work. **

Ik probeerde eerst `git remote add surf` en vervolgens `git fetch`. Toen had ik veel merge conflicts. Ik deed in mijn terminal `git status` en zag ik veel merge conflicts had. Ik probeerde het op te lossen de terminal eerst met `git mergetool`, maar ik wist niet precies hoe dat werkte. Daarom had ik het opgelost in visual studio code, maar ik had geen idee of ik het goed had gedaan. Ik had namelijk alle incoming geaccepteerd. Toen wilde ik checken of alle files van https://github.com/SURF-ML/MLOps_2026 overeenkwam met onze repository. Dus ik dacht dan doe ik opnieuw git fetch surf, maar het was "already up to date" . Toen probeerde ik git push origin main --force en nu is mijn eigen werk van assigment 1 weg. 

**Question 7.5 Branching Discipline: Show the output of git log –graph –oneline –all
–max-count=15. Does your history show clear branching and merging, or is it a
single "flat" line of commits? Would we prefer a non-linear or a linear graph for team
collaboration?**