import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#### GENERATING DATA#####
# Data generation form simple linear model with censoring at random points. Each censoring may be at different value.
# If we observe a value but we know that the real value is more or less than the observed one. We call it censoring

NUM_SAMPLES = 3000

# Linear generative model parameteres
B = [5, -2, 3]
SIGMA = 5

input_size = len(B)
# y_train is latent variable
x_train = np.random.randn(NUM_SAMPLES, input_size)*100
y_train = np.sum(x_train * B, axis=1) + np.random.randn(NUM_SAMPLES) * SIGMA

y_train = y_train.reshape(NUM_SAMPLES, -1)

# creating hypothethical censor points
censor_train = y_train + np.random.randn(NUM_SAMPLES, 1)*5
# determine is it censoring from below or from abovee
down_censor = (censor_train > y_train).astype(int)
up_censor = (censor_train < y_train).astype(int)
# selecting wich observations are actually censored.
censored_ind = np.random.choice(2, (NUM_SAMPLES, 1))
down_censor = censored_ind * down_censor
up_censor = censored_ind * up_censor
# creating observable labels
y_train[censored_ind == 1] = censor_train[censored_ind == 1]


# final dataset consist of:
# y_train - observable variable
# down_censor - indicator of censoring from below (also known as left censoring)
# up_censor - indicator of censoring from above (also known as right censoring)

class LinearRegression(torch.nn.Module):
    def __init__(self, number_of_parameters):
        super().__init__()
        self.b = torch.nn.Parameter(torch.randn((number_of_parameters)))
        self.bias = torch.nn.Parameter(torch.randn(()))
        self.sigma = torch.nn.Parameter(
            torch.abs(torch.randn(()))
        )

    def forward(self, x):
        return torch.matmul(x, self.b) + self.bias

    def string(self):
        return f'{self.b} {self.sigma} {self.bias}'


zero_tensor = torch.tensor([0.0]).to(device)


def loglike(y, sigma, y_, up_ind, down_ind):
    """Calculate logarithm of likelihood for censored tobit model.
    Args:
        Model parameters:
            y: model output
            sigma: variance of random error (estimated during learning)
        True data:
            y_: observed data
            up_ind: boolean indication of right censoring
            down_ind: boolean indication of left censoring
    Returns:
        Logharithm of likelihood
    """

    normaldist = torch.distributions.Normal(
        zero_tensor, sigma)

    # model outputs normal distribution with center at y and std at sigma

    # probability function of normal distribution at point y_
    not_censored_log = normaldist.log_prob(y_ - y)
    # probability of point random variable being more than y_
    up_censored_log_argument = (1 - normaldist.cdf(y_ - y))
    # probability of random variable being less than y_
    down_censored_log_argument = normaldist.cdf(y_ - y)

    up_censored_log_argument = torch.clip(
        up_censored_log_argument, 0.00001, 0.99999)
    down_censored_log_argument = torch.clip(
        down_censored_log_argument, 0.00001, 0.99999)

    # logarithm of likelihood
    loglike = not_censored_log * (1 - up_ind) * (1 - down_ind)
    loglike += torch.log(up_censored_log_argument) * up_ind
    loglike += torch.log(down_censored_log_argument) * down_ind

    loglike2 = torch.sum(loglike)
    # we want to maximize likelihood, but optimizer minimizes by default
    loss = -loglike2
    return loss


model = LinearRegression(3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

x_train = torch.from_numpy(x_train).float().to(device)
y_train = torch.from_numpy(y_train).float().squeeze().to(device)
up_censor = torch.from_numpy(up_censor).squeeze().to(device)
down_censor = torch.from_numpy(down_censor).squeeze().to(device)

my_dataset = TensorDataset(x_train, y_train, up_censor, down_censor)
my_dataloader = DataLoader(my_dataset, batch_size=128, shuffle=True)

for t in range(500):
    for x_train_batch, y_train_batch, up_censor_batch, down_censor_batch in my_dataloader:
        y_pred = model(x_train_batch)
        loss = loglike(y_pred, model.sigma, y_train_batch,
                       up_censor_batch, down_censor_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if t % 100 == 0:
        print(t, loss.item())

print(f'Result: {model.string()}')
