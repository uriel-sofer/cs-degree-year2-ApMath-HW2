import numpy as np

def normip(v, p):
    """
    function to compute the natural norm of an input vector.
    Inputs: v - a numpy array (n dim vector), p – real number >=1,
    if p = ‘max’ the infinity norm is computed (norma ∞).
    Outputs: p norm of v
    :param v: a numpy array (n dim vector)
    :param p: p – real number >=1 or 'max'
    :return: p norm of v
    """

    if p == 'max':
        return np.max(np.abs(v))

    elif p >= 1:
        return np.sum(np.abs(v)**p)** (1 / p)

    else:
        raise ValueError('p must be either real number or "max"')

def alef():
    v = np.array([1j, 2j, -3, 1, 7-3j])
    print("Norm 2 for vector v: ", normip(v, 2))

def beit():
    v = np.array([1j, 5, 2-3j, -1+1j, 2j])
    norm_v = normip(v, 2)
    print("Unit vector for b:" ,v / norm_v)


def gimel():
    u = np.array([2, 1, -3j, 3, 9])
    v = np.array([6j, 7, 2.2j, 7, 0])

    diff = u - v
    p2_norm = normip(diff, 2)
    inf_norm = normip(diff, "max")

    print("Distance (Euclidian norm): ", p2_norm)
    print("Infinity norm: ", inf_norm)

def main():
    alef()
    beit()
    gimel()

if __name__ == '__main__':
    main()