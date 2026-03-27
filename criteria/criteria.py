import numpy as np

def pesimisticke(matrix, info=False):
    G_i = np.min(matrix, axis=1)
    best_decision_index = np.argmax(G_i)

    if info:
        print("--- Pesimistické (Waldovo) kritérium ---")
        print(f"Minimá po riadkoch (G_i): {G_i}")

    return best_decision_index

def optimicke(matrix, info=False):
    G_i = np.max(matrix, axis=1)
    best_decision_index = np.argmax(G_i)

    if info:
        print("--- Optimistické kritérium ---")
        print(f"Maximá po riadkoch (G_i): {G_i}")

    return best_decision_index

def hurwiczovo(matrix, alpha=0.5, info=False):
    if alpha < 0 or alpha > 1:
        print('Alpha must be between 0 and 1!')
        return

    G_i = alpha*np.max(matrix, axis=1) + (1 - alpha)*np.min(matrix, axis=1)
    best_decision_index = np.argmax(G_i)

    if info:
        print(f"--- Hurwiczovo kritérium (alpha={alpha}) ---")
        print(f"Hodnoty G_i pre každý riadok: {G_i}")

    return best_decision_index

def savageovo(matrix, info=False):
    max_in_columns = np.max(matrix, axis=0)
    regret_matrix = max_in_columns - matrix
    G_i = np.max(regret_matrix, axis=1)
    best_decision_index = np.argmin(G_i)

    if info:
        print("--- Savageovo (minimax regret) kritérium ---")
        print(f"Maximá v stĺpcoch: {max_in_columns}")
        print(f"Matica ľútosti (Regret matrix):\n{regret_matrix}")
        print(f"Maximálne straty po riadkoch (G_i): {G_i}")

    return best_decision_index

def bayesovo(matrix, probabilities, info=False):
    G_i = np.sum(matrix * probabilities, axis=1)
    best_decision_index = np.argmax(G_i)

    if info:
        print("--- Bayesovo kritérium ---")
        print(f"Očakávané zisky po riadkoch (G_i): {G_i}")

    return best_decision_index

def laplaceovo(matrix, info=False):
    n = len(matrix[0])
    probabilities = [1 / n] * n
    return bayesovo(matrix, probabilities, info)

def hayashiho(matrix, info=False):
    min_in_columns = np.min(matrix, axis=0)
    g_ij = matrix - min_in_columns
    G_i = np.min(g_ij, axis=1)
    best_decision_index = np.argmax(G_i)

    if info:
        print("--- Hayashiho kritérium ---")
        print(f"Minimá v stĺpcoch: {min_in_columns}")
        print(f"Upravená matica g_ij:\n{g_ij}")
        print(f"Minimá po riadkoch (G_i): {G_i}")

    return best_decision_index

def ocakavanej_strarty_prilezitosti(matrix, probabilities, info=False):
    regret_matrix = np.max(matrix, axis=0) - matrix
    G_i = np.sum(regret_matrix * probabilities, axis=1)
    best_decision_index = np.argmin(G_i)

    if info:
        print("--- Kritérium očakávanej straty príležitosti ---")
        print(f"Matica ľútosti (Regret matrix):\n{regret_matrix}")
        print(f"Očakávané straty po riadkoch (G_i): {G_i}")

    return best_decision_index

def hodges_lehmannovo(matrix, alpha, probabilities, info=False):
    if alpha < 0 or alpha > 1:
        print('Alpha must be between 0 and 1!')
        return
    G_i = alpha*np.min(matrix, axis=1) + (1 - alpha)*np.sum(matrix*probabilities, axis=1)
    best_decision_index = np.argmax(G_i)

    if info:
        print(f"--- Hodges-Lehmannovo kritérium (alpha={alpha}) ---")
        print(f"Hodnoty G_i pre každý riadok: {G_i}")

    return best_decision_index

def hurwics_savageovo(matrix, alpha, info=False):
    if alpha < 0 or alpha > 1:
        print('Alpha must be between 0 and 1!')
        return

    regret_matrix = np.max(matrix, axis=0) - matrix
    G_i = alpha*np.min(matrix, axis=1) + (1 - alpha)*np.max(regret_matrix, axis=1)
    best_decision_index = np.argmin(G_i)

    if info:
        print(f"--- Hurwicz-Savageovo kritérium (alpha={alpha}) ---")
        print(f"Matica ľútosti:\n{regret_matrix}")
        print(f"Hodnoty G_i po riadkoch: {G_i}")

    return best_decision_index

def hermeyerovo(matrix, probabilities, info=False):
    G_i = np.min(matrix*probabilities, axis=1)
    best_decision_index = np.argmax(G_i)

    if info:
        print("--- Hermeyerovo kritérium ---")
        print(f"Matica prenásobená pravdepodobnosťami:\n{matrix * probabilities}")
        print(f"Minimá po riadkoch (G_i): {G_i}")

    return best_decision_index

def najpravdepodobnejsi_stav(matrix, probabilities, info=False):
    k = np.argmax(probabilities)
    best_decision_index = np.argmax(matrix[:, k])
    g_r = matrix[best_decision_index, k]

    if info:
        print("--- Kritérium najpravdepodobnejšieho stavu ---")
        print(f"Najpravdepodobnejší stav (index stĺpca k): {k} (Pravdepodobnosť: {probabilities[k]})")
        print(f"Hodnoty v tomto stĺpci: {matrix[:, k]}")
        print(f"Maximálna hodnota (g_r): {g_r}")

    return best_decision_index



def starrovo_domenove(matrix, num_samples=100000, info=False):
    # Vzhľadom na zložitosť implementácie matematických výpočtov
    # sa na výber najlepšieho riešenia používa metóda Monte Carlo


    m, n = matrix.shape

    # Generovanie náhodných pravdepodobnostných vektorov
    random_points = np.random.exponential(scale=1.0, size=(num_samples, n))
    random_points /= random_points.sum(axis=1, keepdims=True)

    # Vypočítame očakávané výhry G_i pre všetky náhodné rozdelenia pravdepodobnosti
    expected_values = np.dot(matrix, random_points.T)

    # Pre každý bod nájdeme riešenie, ktoré dáva maximum
    best_decisions_for_samples = np.argmax(expected_values, axis=0)

    # Pre každé riešenie vypočítame počet "výhier" (objem domény D_i)
    domain_volumes = np.bincount(best_decisions_for_samples, minlength=m)
    domain_fractions = domain_volumes / num_samples

    best_decision_index = np.argmax(domain_volumes)
    if info:
        print(f"--- Starrovo doménové kritérium (vzoriek: {num_samples}) ---")
        print(f"Počet 'víťazstiev' pre jednotlivé riadky: {domain_volumes}")
        print(f"Plošný obsah domén (podiel): {domain_fractions}")

    return best_decision_index

def sucinove(matrix, info=False):
    min_val = np.min(matrix)
    if min_val <= 0:
        matrix = matrix + np.abs(min_val) + 1

    G_i = np.prod(matrix, axis=1)
    best_decision_index = np.argmax(G_i)

    if info:
        print("--- Súčinové kritérium ---")
        if min_val <= 0:
            print(f"Matica obsahovala nekladné prvky. Upravená matica (+{np.abs(min_val) + 1}):\n{matrix}")
        print(f"Súčiny po riadkoch (G_i): {G_i}")

    return best_decision_index

def apo(matrix, alpha, info=False):
    m, n = matrix.shape
    C = int(max(1, np.ceil(n * min(alpha, 1 - alpha))))
    sorted_matrix = np.sort(matrix, axis=1)[:, ::-1]
    A_max = np.mean(sorted_matrix[:, :C], axis=1)
    A_min = np.mean(sorted_matrix[:, -C:], axis=1)
    H_i = alpha * A_max + (1 - alpha) * A_min
    best_decision_index = np.argmax(H_i)

    if info:
        print(f"--- Modifikované Hurwiczovo (APO) kritérium (alpha={alpha}) ---")
        print(f"Konštanta C: {C}")
        print(f"Zotriedená matica (klesajúco):\n{sorted_matrix}")
        print(f"A_max (priemer {C} najväčších): {A_max}")
        print(f"A_min (priemer {C} najmenších): {A_min}")
        print(f"Výsledné hodnoty H_i: {H_i}")

    return best_decision_index

def cumulative_max_gaspars_wieloch(matrix, info=False):
    m = matrix.shape[0]
    col_sums = np.sum(matrix, axis=0)
    T_c = (m * matrix) - col_sums
    G_c_i = np.min(T_c, axis=1)
    best_decision_index = np.argmax(G_c_i)

    if info:
        print("--- Kumulatívne maximinové kritérium radosti ---")
        print(f"Súčty v stĺpcoch: {col_sums}")
        print(f"Nová matica T^C (kumulatívnych ziskov):\n{T_c}")
        print(f"Minimá po riadkoch (G^C_i): {G_c_i}")

    return best_decision_index

def dominant_radosti(matrix, info=False):
    m, n = matrix.shape

    P = np.zeros((m, n), dtype=int)
    for j in range(n):
        column = matrix[:, j]
        for i in range(m):
            current_element = column[i]
            P[i, j] = np.sum(column >= current_element)

    T_d = m - P
    G_d_i = np.sum(T_d, axis=1)
    best_decision_index = np.argmax(G_d_i)

    if info:
        print("--- Dominančné kritérium radosti ---")
        print(f"Matica indexov p^j(g_ij) [P]:\n{P}")
        print(f"Matica T^D (m - P):\n{T_d}")
        print(f"Súčty po riadkoch (G^D_i): {G_d_i}")

    return best_decision_index
