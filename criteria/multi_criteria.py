import numpy as np
from scipy.stats import rankdata
import itertools

def metoda_entropie(matrix, info=False):
    matrix = np.array(matrix, dtype=float)
    m, n = matrix.shape

    col_sums = np.sum(matrix, axis=0)
    p = matrix / col_sums

    E = -(1 / np.log(m)) * np.sum(p * np.log(p), axis=0)

    D = 1 - E
    w = D / np.sum(D)

    if info:
        print(f"Stĺpcové súčty: {col_sums}")
        print(f"Entropia kritérií (E): {E}")
    return w


def metoda_poradia(ranks, info=False):
    ranks = np.array(ranks)
    n = len(ranks)

    v = n - ranks + 1

    w = v / (n * (n + 1) / 2)

    if info:  print(f"Priradené váhy (v): {v}")
    return w


def fullerova_metoda(frequencies, modified=False, info=False):
    f = np.array(frequencies)
    n = len(f)

    total_pairs = n * (n - 1) / 2

    if not modified:
        w = f / total_pairs
    else:
        w = (f + 1) / (total_pairs + n)

    if info: print(f"Celkový počet porovnaní: {total_pairs}")
    return w


def bodovacia_metoda_vahy(body):
    body = np.array(body)
    w = body / np.sum(body)
    return w


def saatyho_metoda_vahy(S, info=False):
    S = np.array(S)
    n = S.shape[0]

    row_products = np.prod(S, axis=1)
    geom_means = row_products ** (1 / n)
    w = geom_means / np.sum(geom_means)

    if info: print(f"Geometrické priemery riadkov: {geom_means}")
    return w


def postupny_rozvrh_vah(skupiny_vahy, kriteria_v_skupinach_vahy, info=False):

    vysledne_vahy = []

    for i, vaha_skupiny in enumerate(skupiny_vahy):
        vahy_kriterii = np.array(kriteria_v_skupinach_vahy[i])
        if info: print(f"Skupina {i + 1}: váha {vaha_skupiny}, váhy kritérií {vahy_kriterii}")
        vysledne_vahy.extend(vaha_skupiny * vahy_kriterii)

    return np.array(vysledne_vahy)


def konjunktivna_metoda(matrix, aspiracne_urovne, maximalizacne=True):
    matrix = np.array(matrix)
    aspiracne_urovne = np.array(aspiracne_urovne)

    if maximalizacne:
        splnene = matrix >= aspiracne_urovne
    else:
        splnene = matrix <= aspiracne_urovne

    akceptovatelne_indexy = np.where(np.all(splnene, axis=1))[0]
    return akceptovatelne_indexy


def disjunktivna_metoda(matrix, aspiracne_urovne, maximalizacne=True):
    matrix = np.array(matrix)
    aspiracne_urovne = np.array(aspiracne_urovne)

    if maximalizacne:
        splnene = matrix >= aspiracne_urovne
    else:
        splnene = matrix <= aspiracne_urovne

    akceptovatelne_indexy = np.where(np.any(splnene, axis=1))[0]
    return akceptovatelne_indexy


def metoda_priam_krok(matrix, z_nove, predchadzajuce_platne_indexy=None, idealny_variant=None, maximization=None, info=False):
    matrix = np.array(matrix, dtype=float)
    z_nove = np.array(z_nove, dtype=float)
    m, n = matrix.shape

    if maximization is None: maximization = [True] * n
    if predchadzajuce_platne_indexy is None: predchadzajuce_platne_indexy = np.arange(m)

    splnene = np.zeros((len(predchadzajuce_platne_indexy), n), dtype=bool)
    for idx, orig_idx in enumerate(predchadzajuce_platne_indexy):
        for j in range(n):
            if maximization[j]:
                splnene[idx, j] = matrix[orig_idx, j] >= z_nove[j]
            else:
                splnene[idx, j] = matrix[orig_idx, j] <= z_nove[j]

    if info: print(f"Kontrola aspirácií {z_nove}: splnené={splnene}")

    presli_indexy_lokalne = np.where(np.all(splnene, axis=1))[0]
    presli_indexy_globalne = predchadzajuce_platne_indexy[presli_indexy_lokalne]

    if len(presli_indexy_globalne) == 1:
        return presli_indexy_globalne[0], "Najdeny jediny kompromisny variant", None
    elif len(presli_indexy_globalne) > 1:
        return presli_indexy_globalne, "Je potrebne zvysit aspiracne urovne (krok s+1)", None
    else:
        idealny_variant = np.array(idealny_variant, dtype=float)
        odchylky = [np.sum(np.abs(z_nove - matrix[idx]) / idealny_variant) for idx in predchadzajuce_platne_indexy]
        if info: print(f"Odchýlky: {odchylky}")
        best_idx = predchadzajuce_platne_indexy[np.argmin(odchylky)]
        return best_idx, "Vybrany variant s min. odchylkou", odchylky

def metoda_poradia_varianty(matrix, weights=None, maximization=None):
    matrix = np.array(matrix, dtype=float)
    m, n = matrix.shape

    if weights is None:
        weights = np.ones(n)
    if maximization is None:
        maximization = [True] * n

    rank_matrix = np.zeros((m, n))

    for j in range(n):
        col = matrix[:, j]
        if maximization[j]:
            col = -col

        rank_matrix[:, j] = rankdata(col)
    weighted_ranks = np.sum(rank_matrix * weights, axis=1)

    best_decision_index = np.argmin(weighted_ranks)

    return best_decision_index, rank_matrix, weighted_ranks


def lexikograficka_metoda(matrix, poradie_kriterii, maximization=None):
    matrix = np.array(matrix, dtype=float)
    m, n = matrix.shape

    if maximization is None:
        maximization = [True] * n

    platne_varianty = np.arange(m)

    for krit_idx in poradie_kriterii:
        if len(platne_varianty) == 1:
            break

        hodnoty = matrix[platne_varianty, krit_idx]

        if maximization[krit_idx]:
            najlepsia_hodnota = np.max(hodnoty)
        else:
            najlepsia_hodnota = np.min(hodnoty)

        zhoda = np.isclose(hodnoty, najlepsia_hodnota)
        platne_varianty = platne_varianty[zhoda]

    return platne_varianty


def bodovacia_metoda_varianty(scored_matrix, weights):
    matrix = np.array(scored_matrix, dtype=float)
    weights = np.array(weights, dtype=float)

    G_i = np.sum(matrix * weights, axis=1)

    best_decision_index = np.argmax(G_i)
    return best_decision_index, G_i

def metoda_vazeneho_suctu(matrix, weights, maximization=None):
    matrix = np.array(matrix, dtype=float)
    m, n = matrix.shape
    weights = np.array(weights)

    if maximization is None:
        maximization = [True] * n

    U = np.zeros((m, n))

    for j in range(n):
        col = matrix[:, j]

        if maximization[j]:
            h_j = np.max(col)
            d_j = np.min(col)
        else:
            h_j = np.min(col)
            d_j = np.max(col)

        if h_j == d_j:
            U[:, j] = 1.0
        else:
            U[:, j] = (col - d_j) / (h_j - d_j)

    G_i = np.sum(U * weights, axis=1)

    best_decision_index = np.argmax(G_i)

    return best_decision_index, U, G_i


def permutacna_metoda(matrix, weights, maximization=None):
    matrix = np.array(matrix, dtype=float)
    weights = np.array(weights, dtype=float)
    m, n = matrix.shape

    if maximization is None:
        maximization = [True] * n

    C = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i != j:
                c_ij = 0
                for k in range(n):
                    if maximization[k]:
                        if matrix[i, k] >= matrix[j, k]:
                            c_ij += weights[k]
                    else:
                        if matrix[i, k] <= matrix[j, k]:
                            c_ij += weights[k]
                C[i, j] = c_ij

    best_R = -np.inf
    best_permutation = None

    for perm in itertools.permutations(range(m)):
        C_phi = C[np.ix_(perm, perm)]

        sum_upper = np.sum(np.triu(C_phi, k=1))
        sum_lower = np.sum(np.tril(C_phi, k=-1))
        R_phi = sum_upper - sum_lower

        if R_phi > best_R:
            best_R = R_phi
            best_permutation = perm

    return best_permutation, best_R, C


def metoda_ahp_varianty(saaty_kriteria, zoznam_saaty_varianty):

    def get_weights(S):
        n = S.shape[0]
        geom_means = np.prod(S, axis=1) ** (1 / n)
        return geom_means / np.sum(geom_means)

    w_criteria = get_weights(np.array(saaty_kriteria))
    n = len(w_criteria)
    m = np.array(zoznam_saaty_varianty[0]).shape[0]

    variant_weights_matrix = np.zeros((m, n))
    for j in range(n):
        w_var = get_weights(np.array(zoznam_saaty_varianty[j]))
        variant_weights_matrix[:, j] = w_var

    final_scores = np.sum(variant_weights_matrix * w_criteria, axis=1)

    best_decision_index = np.argmax(final_scores)

    return best_decision_index, final_scores, variant_weights_matrix


def metoda_topsis(matrix, weights, maximization=None):
    matrix = np.array(matrix, dtype=float)
    weights = np.array(weights, dtype=float)
    m, n = matrix.shape

    if maximization is None:
        maximization = [True] * n

    norm_divisors = np.sqrt(np.sum(matrix ** 2, axis=0))
    norm_divisors[norm_divisors == 0] = 1.0
    R = matrix / norm_divisors

    Z = R * weights

    h = np.zeros(n)
    d = np.zeros(n)

    for j in range(n):
        if maximization[j]:
            h[j] = np.max(Z[:, j])
            d[j] = np.min(Z[:, j])
        else:
            h[j] = np.min(Z[:, j])
            d[j] = np.max(Z[:, j])

    d_plus = np.sqrt(np.sum((Z - h) ** 2, axis=1))

    d_minus = np.sqrt(np.sum((Z - d) ** 2, axis=1))

    denominator = d_plus + d_minus
    c_i = np.zeros(m)

    valid_idx = denominator > 0
    c_i[valid_idx] = d_minus[valid_idx] / denominator[valid_idx]

    best_decision_index = np.argmax(c_i)

    return best_decision_index, c_i, d_plus, d_minus

def metoda_oreste_hodnotenie(q_ranks, P_matrix, r=3):
    q = np.array(q_ranks, dtype=float)
    P = np.array(P_matrix, dtype=float)
    m, n = P.shape

    D = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            D[i, j] = ((P[i, j]**r) / 2 + (q[j]**r) / 2) ** (1/r)

    R = rankdata(D).reshape(m, n)

    r_sums = np.sum(R, axis=1)

    best_decision_index = np.argmin(r_sums)

    return best_decision_index, r_sums, R, D

def metoda_electre_I(matrix, weights, c_star, d_star, maximization=None):
    matrix = np.array(matrix, dtype=float)
    weights = np.array(weights, dtype=float)
    m, n = matrix.shape

    if maximization is None:
        maximization = [True] * n

    C_idx = np.zeros((m, m))
    D_idx = np.zeros((m, m))
    P = np.zeros((m, m), dtype=int)

    max_diffs = np.zeros(n)
    for j in range(n):
        max_diffs[j] = np.max(matrix[:, j]) - np.min(matrix[:, j])
    max_global_diff = np.max(max_diffs)

    for i in range(m):
        for j in range(m):
            if i == j: continue

            c_ij = 0
            max_d_ij = 0

            for h in range(n):
                diff = matrix[j, h] - matrix[i, h]
                if (maximization[h] and matrix[i, h] >= matrix[j, h]) or \
                   (not maximization[h] and matrix[i, h] <= matrix[j, h]):
                    c_ij += weights[h]
                else:
                    if abs(diff) > max_d_ij:
                        max_d_ij = abs(diff)

            C_idx[i, j] = c_ij
            D_idx[i, j] = max_d_ij / max_global_diff if max_global_diff != 0 else 0

            if C_idx[i, j] >= c_star and D_idx[i, j] <= d_star:
                P[i, j] = 1

    effective_variants = []
    for i in range(m):
        if np.sum(P[:, i]) == 0 and np.sum(P[i, :]) > 0:
            effective_variants.append(i)

    return effective_variants, P, C_idx, D_idx

def metoda_agrepref(matrix, weights, alpha=1.0, beta=0.0, maximization=None):
    matrix = np.array(matrix, dtype=float)
    weights = np.array(weights, dtype=float)
    m, n = matrix.shape

    if maximization is None:
        maximization = [True] * n

    Relations = np.zeros((m, m), dtype=int)

    for i in range(m):
        for j in range(m):
            if i >= j: continue

            s_ij, s_ji, s_eq = 0.0, 0.0, 0.0

            for h in range(n):
                if matrix[i, h] == matrix[j, h]:
                    s_eq += weights[h]
                elif (maximization[h] and matrix[i, h] > matrix[j, h]) or \
                     (not maximization[h] and matrix[i, h] < matrix[j, h]):
                    s_ij += weights[h]
                else:
                    s_ji += weights[h]

            if s_eq >= alpha:
                Relations[i, j] = Relations[j, i] = 2
            elif s_ij - s_ji > beta:
                Relations[i, j] = 1
            elif s_ji - s_ij > beta:
                Relations[j, i] = 1
            else:
                Relations[i, j] = Relations[j, i] = 3

    return Relations