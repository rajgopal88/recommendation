import numpy as np
from scipy import linalg, dot

# this is just a demo array, there will be small changes in this algorithm,
# the whole code will be converted to a function which will have different
# parameters such as array according to the demand of retailer.

def recommendation(R, user_a=1, item_j=2):
    row_sums = R.sum(axis=1)
    column_sums = R.sum(axis=0)

    numrows = len(R)
    numcols = len(R[0])

    row_avg = np.zeros(numrows)
    for i in range(0, numrows):
        row_avg[i] = row_sums[i] / numrows
    # print("Row Avg")
    # print(row_avg)

    col_avg = np.zeros(numcols)
    for i in range(0, numcols):
        col_avg[i] = column_sums[i] / numcols
    # print("Column Avg")
    # print(col_avg)

    R_new = np.zeros(shape=(numrows, numcols))
    for i in range(0, numrows):
        for j in range(0, numcols):
            if (R[i][j] == 0):
                R_new[i][j] = col_avg[j]
            else:
                R_new[i][j] = R[i][j]
                # print("R_new")
    # print(R_new)

    R_nor = np.zeros(shape=(numrows, numcols))
    for i in range(0, numrows):
        for j in range(0, numcols):
            R_nor[i][j] = R_new[i][j] - row_avg[i]
    # print("R_nor")
    # print(R_nor)

    U, S, V = linalg.svd(R_nor, full_matrices=1, compute_uv=1)
    # print("U     S      V")
    # print(U), print(S), print(V)

    R_red = dot(dot(U, linalg.diagsvd(S, len(S), len(V))), V)
    # print("R_red")
    # print(R_red)

    U_n, S_n, V_n = linalg.svd(R_red, full_matrices=1, compute_uv=1)
    # print("U_n      S_n       V_n")
    # print(U_n), print(S_n), print(V_n)

    dimensions = 1
    if dimensions <= numrows:
        for index in range(numrows - dimensions, numrows):
            S[index] = 0

    # reducing dimensionality of S
    S_k = np.zeros((numrows - 1, numrows - 1))
    for i in range(0, numrows - 1):
        S_k[i][i] = S_n[i]

    # reducing dimensionality of U
    U_k = np.zeros((numrows, numrows - 1))
    for i in range(0, numrows):
        for j in range(0, numrows - 1):
            U_k[i][j] = U_n[i][j]

    # reducing dimensionality of V

    # print("V_n")
    # print(V_n)
    V_k = np.zeros((numrows - 1, numcols))
    for i in range(0, numrows - 1):
        for j in range(0, numcols):
            V_k[i][j] = V_n[i][j]

    # print("U_k     S_k       V_k")
    # print(U_k), print(S_k), print(V_k)

    sq_S_k = linalg.sqrtm(S_k)
    # print("Sq_S_k")
    # print(sq_S_k)

    # print(len(U_k)), print(len(U_k[0])), print(len(S_k)), print(len(S_k[0])), print(len(V_k)), print(len(V_k[0]))
    A = np.matmul(U_k, sq_S_k.transpose())
    B = np.matmul(sq_S_k, V_k)
    # print("BB")
    # print(B)
    row = len(B)
    col = len(B[0])

    # this is not reuired but laterthis is just for testing later should be removed
    R_new = np.zeros((row, col))
    # print(row), print(col)
    for j in range(0, row):
        for f in range(0, col):
            # print(adjusted_cosine_similarity(B, j, f))
            R_new[j][f] = adjusted_cosine_similarity(B, j, f)

    print("R_new", R_new)

    pr = 0
    u = 0
    v = 0
    # l is not found yet needs to be clarified by alix
    l = [1.02, 2.03, 3.04, 5.06]
    for k in l:
        u = u + adjusted_cosine_similarity(B, item_j, k) * (R_red[user_a][item_j] * row_avg[user_a])
        v = v + np.absolute(adjusted_cosine_similarity(B, item_j, k))
        pr = pr + u / v

        # print(pr)
        # result  = prediction(B, user_a=1, item_j=2)
        # return result


def adjusted_cosine_similarity(array, j, f):
    item1 = 0
    item2 = 0
    item3 = 0
    for i in range(0, len(array)):
        item1 = item1 + (array[i][j] * array[i][f])
        item2 = item2 + (array[i][j] * array[i][j])
        item3 = item3 + (array[i][f] * array[i][f])
    result = item1 / (np.sqrt(item2 * item3) * np.sqrt(item3 * item3))
    return result


def prediction(B, user_a, item_j):
    pr = 0
    u = 0
    v = 0
    # l is not found yet needs to be clarified by alix
    l = [1.02, 2.03, 3.04, 5.06]
    for k in l:
        u = u + adjusted_cosine_similarity(B, item_j, k) * (R_red[user_a][item_j] * row_avg[user_a])
        v = v + np.absolute(adjusted_cosine_similarity(B, item_j, k))
        pr = pr + u / v
    return pr


if __name__ == "__main__":
    R = np.asarray([[1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 1, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 1, 1, 0, 1, 1, 0, 0, 0, 0]])

    recommendation(R, user_a=1, item_j=2)
