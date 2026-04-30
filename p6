import numpy as np
import matplotlib.pyplot as plt

def takhmin_ehtemal_behtarin(alpha_ha, beta_ha, n_samples=10000):
    tedad_bazoo = len(alpha_ha)
    bord_ha = np.zeros(tedad_bazoo)
    
    for s in range(n_samples):
        nemone_ha = []
        for i in range(tedad_bazoo):
            adad = np.random.beta(alpha_ha[i], beta_ha[i])
            nemone_ha.append(adad)
        
        behtarin_kodome = np.argmax(nemone_ha)
        bord_ha[behtarin_kodome] = bord_ha[behtarin_kodome] + 1
        
    return bord_ha / n_samples

T_list = [500, 1000, 5000]
mu_vaghei = [0.2, 0.4, 0.6, 0.3, 0.25]
tedad_bazoo = len(mu_vaghei)

for T in T_list:
    alpha_ha = np.ones(tedad_bazoo)
    beta_ha = np.ones(tedad_bazoo)
    dafat_entekhab = np.zeros(tedad_bazoo)
    
    for t in range(T):
        nemone_ts = []
        for i in range(tedad_bazoo):
            adad = np.random.beta(alpha_ha[i], beta_ha[i])
            nemone_ts.append(adad)
        
        entekhab = np.argmax(nemone_ts)
        dafat_entekhab[entekhab] = dafat_entekhab[entekhab] + 1
        
        shans = np.random.random()
        if shans < mu_vaghei[entekhab]:
            alpha_ha[entekhab] = alpha_ha[entekhab] + 1
        else:
            beta_ha[entekhab] = beta_ha[entekhab] + 1
            
    ehtemal_takhmini = takhmin_ehtemal_behtarin(alpha_ha, beta_ha)
    faravani_tajrobi = dafat_entekhab / T
    
    ham_bastegi = np.corrcoef(ehtemal_takhmini, faravani_tajrobi)[0, 1]
    
    print("Results T =", T)
    print("Estimated Probabilities:", ehtemal_takhmini)
    print("Selection Frequencies:", faravani_tajrobi)
    print("Correlation:", ham_bastegi)
  

    plt.figure()
    plt.bar(range(1, tedad_bazoo + 1), ehtemal_takhmini, alpha=0.5, label="Monte Carlo Prob")
    plt.bar(range(1, tedad_bazoo + 1), faravani_tajrobi, alpha=0.5, label="Selection Freq")
    plt.title("T = " + str(T))
    plt.xlabel('step')
    plt.ylabel("cumulative regret")
    plt.legend()
    plt.show()
