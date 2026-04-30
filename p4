import numpy as np
import matplotlib.pyplot as plt

T = 10000
tedad_bazoo = 5
mu_vaghei = [0.15, 0.3, 0.45, 0.6, 0.8]
behtarin_mu = max(mu_vaghei)

alpha_ha = np.ones(tedad_bazoo)
beta_ha = np.ones(tedad_bazoo)

regret_kolli = 0
list_regret = []
tarikhche_mu = []
tarikhche_variance = []

for t in range(T):
    nemone_ha = []
    for i in range(tedad_bazoo):
        adad = np.random.beta(alpha_ha[i], beta_ha[i])
        nemone_ha.append(adad)
    
    entekhab = np.argmax(nemone_ha)
    
    shans = np.random.random()
    if shans < mu_vaghei[entekhab]:
        reward = 1
    else:
        reward = 0
    
    regret_kolli = regret_kolli + (behtarin_mu - mu_vaghei[entekhab])
    list_regret.append(regret_kolli)
    
    if reward == 1:
        alpha_ha[entekhab] = alpha_ha[entekhab] + 1
    else:
        beta_ha[entekhab] = beta_ha[entekhab] + 1
        
    mu_feli = []
    var_feli = []
    for i in range(tedad_bazoo):
        a = alpha_ha[i]
        b = beta_ha[i]
        miangin = a / (a + b)
        varians = (a * b) / ((a + b)**2 * (a + b + 1))
        mu_feli.append(miangin)
        var_feli.append(varians)
    
    tarikhche_mu.append(mu_feli)
    tarikhche_variance.append(var_feli)

plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.plot(list_regret)
plt.title("Regret")
plt.subplot(3, 1, 2)
plt.plot(tarikhche_mu)
plt.legend(["Arm 1", "Arm 2", "Arm 3", "Arm 4", "Arm 5"])
plt.title("Means")
plt.subplot(3, 1, 3)
plt.plot(tarikhche_variance)
plt.title("Uncertainty")
plt.tight_layout()
plt.show()
