    ### Tasks
1. Convert the existing code to support multiple action environement

## Features
The current implementation follows the 9 steps that are mentioned in the lecture somewhat summarized below
1. Do batch updates
2. Do multple epochs on a single batch
3. Remove the \gamma^t term from the policy loss
4. Use lambda returns for all calculation of returns and activateions
5. Use advanttage calucaluation forw here you negate teh value from the return
6. Normalize the advantage for the batch
7. Add the proximity constraints on the importnace sampling


### MOdification Left
1. ***Normalize Observations*** : Normalize the observations by keeping a running means and running standard deviation, according to the baselines implementation they have have a running mean and standar deviation and theyu clip the oibservation values between [ 0.5, 0.5]
2. ***Value Loss Coeff*** : Have a value loss coeff to the combined loss of the policy and the value.
3. ***Normalize the rewards*** : Normalize the rewartds
4. ***Clip all the gradients"***
5. ***Value Clipping*** : Clip the value doifference.


Problems : 
Hmm , it seems that the sacle action false is not working
sedcoiind, need to test the noramlization of th observatio spac eand also need to clips the observations between -0.5, 0.5