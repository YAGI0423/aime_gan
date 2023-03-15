import argparse
from torch.utils.data import DataLoader
from logitGateDataset.datasets import AndGate, OrGate, XorGate, NotGate


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Argument Help')
    parser.add_argument('--logit', type=str, default='XOR', choices=('AND', 'OR', 'XOR'))
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--dataset_size', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=6)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    dataset_args = {
        'dataset_size': args.dataset_size,
        'input_size': args.input_size,
    }

    if args.logit == 'AND':
        dataset = AndGate(**dataset_args)
    elif args.logit == 'OR':
        dataset = OrGate(**dataset_args)
    elif args.logit == 'XOR':
        dataset = XorGate(**dataset_args)
    elif args.logit == 'NOT':
        dataset = NotGate(**dataset_args)

    dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    x, y = next(iter(dataLoader))
    print(x)

    print(y)