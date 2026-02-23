def compute_specificity(true_positive, false_positive):
    if true_positive + false_positive == 0:
        return 0  # To avoid division by zero
    return true_positive / (true_positive + false_positive)

# Example usage
if __name__ == '__main__':
    true_positive = 30
    false_positive = 10
    specificity = compute_specificity(true_positive, false_positive)
    print(f'Specificity: {specificity}')