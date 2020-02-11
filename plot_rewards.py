import numpy
import argparse
import matplotlib.pyplot as pyplot

# reward averages and plot

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, required=True, help='either "train" or "test"')

args = parser.parse_args()

reward = numpy.load(f'linear_rewards/{args.mode}.npy')

print(f"average reward: {reward.mean():.2f}, min: {reward.min():.2f}, max: {reward.max():.2f}")

pyplot.hist(reward, bins=20)
pyplot.title(args.mode)
pyplot.show()