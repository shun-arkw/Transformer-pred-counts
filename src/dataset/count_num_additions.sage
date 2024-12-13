import bisect
import numpy as np
import timeout
# from timeout_decorator import timeout, TimeoutError
timeout_duration = 2

class NumAdditionsCounter:
    def __init__(self, weight_matrix, polynomial_list, coeff_field, num_variables=3, select_strategy='normal', stop_algorithm=True, threshold=10000, timeout=None):

        self.weight_matrix = weight_matrix
        self.polynomial_list = polynomial_list
        self.coeff_field = coeff_field
        self.num_variables = num_variables
        self.select_strategy = select_strategy
        self.stop_algorithm = stop_algorithm
        self.threshold = threshold



    def run(self):
        weight_matrix = self.weight_matrix

        if isinstance(weight_matrix, np.ndarray):
            int_matrix = weight_matrix.astype(int)
        elif isinstance(weight_matrix, str):
            int_matrix = self._str_to_matrix(weight_matrix)
        else:
            raise ValueError('weight_matrix must be np.ndarray or str')



        assert self._validation(int_matrix), "The weight matrix is invalid"

        _matrix = matrix(int_matrix.tolist()) # SageMathのmatrixオブジェクトに変換(引数がnp.ndarrayは不可)

        Ring = PolynomialRing(self.coeff_field, self.num_variables, 'x', order=TermOrder(_matrix))

        polynomial_list = [p for p in self.polynomial_list if p != 0]
        polynomial_list = list(map(Ring, polynomial_list))

        buchberger_algorithm = BuchbergerAlgorithm(Ring, polynomial_list, select_strategy=self.select_strategy, stop_algorithm=self.stop_algorithm, threshold=self.threshold)

        try:
            GB, stat = buchberger_algorithm()
            additions = stat['polynomial_additions']

        except TimeoutError:
            GB = []
            additions = self.threshold + 1

            pass

        if additions > self.threshold:
            # return GB, self.threshold + 1
            return GB, additions
        else:
            return GB, additions

    def _validation(self, weight_matrix):
        num_rows, num_columns = weight_matrix.shape
        assert num_rows == num_columns, "The matrix must be square"

        # The weight matrix must contain integer elements.
        if not np.issubdtype(weight_matrix.dtype, np.integer):
            raise ValueError("The weight matrix must contain integer elements")
        
        # The first non-zero element in each column of the weight matrix must be positive.
        transposed_matrix = weight_matrix.T 
        for row in transposed_matrix:
            for elem in row:
                if elem > 0:
                    break
                elif elem < 0:
                    return False
                
        # The weight matrix must be full rank.
        rank = np.linalg.matrix_rank(weight_matrix)
        if rank == num_rows:
            return True
        else:
            return False 
    
    def _str_to_matrix(self, str_term_order):
        nvars = self.num_variables

        if str_term_order == 'lex':
            return np.eye(nvars, dtype=int)
        
        elif str_term_order == 'grlex' or str_term_order == 'grevlex':

            first_row = np.ones((1, self.num_variables), dtype=int)
            rows = np.eye(nvars, dtype=int)

            if str_term_order == 'grlex':
                weight_matrix = np.concatenate([first_row, rows])
            else: # grevlex
                rows = np.flip(np.eye(nvars, dtype=int), axis=1)
                rows = -rows

            weight_matrix = np.concatenate([first_row, rows])
            weight_matrix = weight_matrix[:nvars, :]

            return weight_matrix
        else:
            raise ValueError("文字列で指定するなら 'lex', 'grlex' or 'grevlex'のみ ")




class BuchbergerAlgorithm:
    def __init__(self, Ring, polynomial_list, S=None, select_strategy='degree', stop_algorithm=True, threshold=100000, timeout=5, elimination='gebauermoeller', rewards='additions', sort_reducers=True, gamma=0.99):
        self.Ring = Ring
        self.polynomial_list = polynomial_list
        self.S = S
        self.select_strategy = select_strategy
        self.stop_algorithm = stop_algorithm
        self.threshold = threshold
        self.elimination = elimination
        self.rewards = rewards
        self.sort_reducers = sort_reducers
        self.gamma = gamma

        self.stats = {'zero_reductions': 0,
                'nonzero_reductions': 0,
                'polynomial_additions': 0,
                'total_reward': 0.0,
                'discounted_return': 0.0}

    # @timeout.timeout(duration=timeout_duration)
    def __call__(self):
        Ring = self.Ring
        F = self.polynomial_list
        S = self.S

        F_new = list(map(Ring, F))

        if S is None:
            G, lmG = [], []
            P = []
            for f in F_new:
                G, P = self.update(G, P, self.monic(f), strategy=self.elimination)
                lmG.append(f.lm())
        
        else:
            G, lmG = F_new, [f.lm() for f in F_new]
            P = S
        
        # stats = {'zero_reductions': 0,
        #         'nonzero_reductions': 0,
        #         'polynomial_additions': 0,
        #         'total_reward': 0.0,
        #         'discounted_return': 0.0}
        discount = 1.0

        if self.sort_reducers and len(G) > 0:
            G_ = G.copy()
            G_.sort(key=lambda g: g.lm())
            lmG_, keysG_ = [g.lm() for g in G_], [self.order(g.lm()) for g in G_]
        else:
            G_, lmG_ = G, lmG

        while P:
            i, j = self.select(G, P)
            P.remove((i, j))

            spoly_ = self.spoly(G[i], G[j], lmf=lmG[i], lmg=lmG[j])
            self.stats['polynomial_additions'] += 1
            
            r, s = self.reduce(spoly_, G_, interrupt_flag=True)

            reward = (-1.0 - s['steps']) if self.rewards == 'additions' else -1.0
            self.stats['polynomial_additions'] += s['steps'] 
            self.stats['total_reward'] += reward
            self.stats['discounted_return'] += discount * reward
            discount *= self.gamma

            if self.stop_algorithm and self.stats['polynomial_additions'] > self.threshold:
                return [], self.stats
            
            if r != 0:
                G, P = self.update(G, P, self.monic(r), lmG=lmG, strategy=self.elimination)
                lmG.append(r.lm())
                if self.sort_reducers:
                    key = self.order(r.lm())
                    index = bisect.bisect(keysG_, key)
                    G_.insert(index, self.monic(r))
                    lmG_.insert(index, r.lm())
                    keysG_.insert(index, key)
                else:
                    G_ = G
                    lmG_ = lmG
                self.stats['nonzero_reductions'] += 1
            else:
                self.stats['zero_reductions'] += 1

        Gmin = self.minimalize(G)
        Gred = self.interreduce(Gmin)
        return Gred, self.stats


    def monic(self, f):
        if f == 0:
            raise ZeroDivisionError('Cannot make a zero polynomial monic')
        return f / f.lc()
    
    def select(self, G, P):
        """Select and return a pair from P."""
        assert len(G) > 0, "polynomial list must be nonempty"
        assert len(P) > 0, "pair set must be nonempty"
        R = self.Ring
        strategy = self.select_strategy

        if isinstance(strategy, str):
            strategy = [strategy]

        def strategy_key(p, s):
            """Return a sort key for pair p in the strategy s."""
            if s == 'first':
                return p[1], p[0]
            elif s == 'normal':
                lcm = R.monomial_lcm(G[p[0]].lm(), G[p[1]].lm())
                return lcm
            elif s == 'degree':
                lcm = R.monomial_lcm(G[p[0]].lm(), G[p[1]].lm())
                return lcm.total_degree()
            elif s == 'random':
                return np.random.rand()
            else:
                raise ValueError('unknown selection strategy')

        return min(P, key=lambda p: tuple(strategy_key(p, s) for s in strategy))
    
    def spoly(self, f, g, lmf=None, lmg=None):
       
        lmf = f.lm() if lmf is None else lmf
        lmg = g.lm() if lmg is None else lmg

        R = self.Ring
        lcm = R.monomial_lcm(lmf, lmg)
        s1 = f * R.monomial_quotient(lcm, lmf, coeff=True)
        s2 = g * R.monomial_quotient(lcm, lmg, coeff=True)
        
        return s1 - s2
    
    def reduce(self, g, F, lmF=None, interrupt_flag=False):
        Ring = self.Ring
        monomial_divides = Ring.monomial_divides
        lmF = [f.lm() for f in F] if lmF is None else lmF

        stats = {'steps': 0}
        r = 0 #あまりを0で初期化
        h = g # hは中間被除数

        while h != 0:
            lmh, lch = h.lm(), h.lc()
            found_divisor = False
            # print(f'中間被除数: {h}')
            for f, lmf in zip(F, lmF):
                
                if monomial_divides(lmf, lmh):
                    m = Ring.monomial_quotient(lmh, lmf)
                    h = h - f * m * lch
                    
                    found_divisor = True
                    stats['steps'] += 1

                    if interrupt_flag and self.stop_algorithm:
                        num_additions = self.stats['polynomial_additions'] + stats['steps']

                        if num_additions > self.threshold:
                            return None, stats
                            
                    break

            if not found_divisor:
                r = r + h.lt()
                h = h - h.lt()

        return r, stats
    
    def update(self, G, P, f, strategy='gebauermoeller', lmG=None):
        lmf = f.lm()
        lmG = [g.lm() for g in G] if lmG is None else lmG
        Ring = self.Ring

        lcm = Ring.monomial_lcm
        mul = operator.mul
        div = Ring.monomial_divides  #　割り切るかどうか div(A, B) AがBを割り切るときTrue

        m = len(G)

        if strategy == 'none':
            P_ = [(i, m) for i in range(m)]

        elif strategy == 'lcm':
            P_ = [(i, m) for i in range(m) if lcm(lmG[i], lmf) != mul(lmG[i], lmf)]
        
        elif strategy == 'gebauermoeller':
            def can_drop(p):
                i, j = p
                gam = lcm(lmG[i], lmG[j])
                return div(lmf, gam) and gam != lcm(lmG[i], lmf) and gam != lcm(lmG[j], lmf)
            P[:] = [p for p in P if not can_drop(p)]

            lcms = {}

            for i in range(m):
                lcms.setdefault(lcm(lmG[i], lmf), []).append(i)
            min_lcms = []
            P_ = []

            for gam in sorted(lcms.keys()): 
                if all(not div(m, gam) for m in min_lcms):
                    min_lcms.append(gam)
                    if not any(lcm(lmG[i], lmf) == mul(lmG[i], lmf) for i in lcms[gam]):
                        P_.append((lcms[gam][0], m))
            P_.sort(key=lambda p: p[0])
        
        else:
            raise ValueError('unknown elimination strategy')
        
        G.append(f)
        P.extend(P_)

        return G, P
    
    def order(self, monomial):

        if not monomial.is_monomial():
            return ValueError('not monomial')

        # sagemath(singular)の使用上，weight matrixの成分が実数であるとき，整数に丸め込まれている可能性がある
        weight_matrix = self.Ring.term_order().matrix()
        exponential_vector = vector(monomial.exponents().pop())

        if weight_matrix:
            return weight_matrix * exponential_vector
        else:
            raise ValueError('monomial order is not defined by the weight matrix')
        
    def minimalize(self, G):
        """Return a minimal Groebner basis from arbitrary Groebner basis G."""

        Ring = self.Ring if len(G) > 0 else None
        Gmin = []

        for f in sorted(G, key=lambda h: h.lm()):
            if all(not Ring.monomial_divides(g.lm(), f.lm()) for g in Gmin):
                Gmin.append(f)

        return Gmin
    
    def interreduce(self, G):
        """Return the reduced Groebner basis from minimal Groebner basis G."""
        Gred = []
        for i in range(len(G)):
            g, _ = self.reduce(G[i], G[:i] + G[i+1:])
            Gred.append(self.monic(g))
        return Gred







def validation(weight_matrix):
    """Return True if the weight matrix is valid, False otherwise.
    
    Parameters
    ----------
    weight_matrix : np.ndarray
        正方行列かつ成分が整数必要がある
    
    
    """

    num_rows, num_columns = weight_matrix.shape
    assert num_rows == num_columns, "The matrix must be square"

    # The weight matrix must contain integer elements.
    if not np.issubdtype(weight_matrix.dtype, np.integer):
        raise ValueError("The weight matrix must contain integer elements")
    
    # The first non-zero element in each column of the weight matrix must be positive.
    transposed_matrix = weight_matrix.T 
    for row in transposed_matrix:
        for elem in row:
            if elem > 0:
                break
            elif elem < 0:
                return False
            
    # The weight matrix must be full rank.
    rank = np.linalg.matrix_rank(weight_matrix)
    if rank == num_rows:
        return True
    else:
        return False


def is_groebner(polynomial_ring, polynomial_list):
    '''Check if a list of polynomials is a Groebner basis.
    
    '''

    Ring = polynomial_ring
    I = Ring.ideal(polynomial_list)
    return I.basis_is_groebner()

def cal_groebner_basis(polynomial_ring, polynomial_list):
    '''Compute the Groebner basis of a list of polynomials.
    
    '''

    Ring = polynomial_ring
    I = Ring.ideal(polynomial_list)
    gb = I.groebner_basis()
    gb = sorted(gb,  key = lambda g : g.lm())

    return gb
