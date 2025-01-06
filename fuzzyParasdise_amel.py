import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

service_quality = ctrl.Antecedent(np.arange(0, 11, 1), 'service_quality') 
meal_temperature = ctrl.Antecedent(np.arange(0, 11, 1), 'meal_temperature')  
cleanliness = ctrl.Antecedent(np.arange(0, 11, 1), 'cleanliness')  
customer_satisfaction = ctrl.Consequent(np.arange(0, 11, 1), 'customer_satisfaction') 

service_quality['poor'] = fuzz.trapmf(service_quality.universe, [0, 0, 2, 4])
service_quality['average'] = fuzz.trimf(service_quality.universe, [3, 5, 7])
service_quality['excellent'] = fuzz.trapmf(service_quality.universe, [6, 8, 10, 10])

meal_temperature['cold'] = fuzz.trapmf(meal_temperature.universe, [0, 0, 3, 5])
meal_temperature['warm'] = fuzz.trimf(meal_temperature.universe, [4, 6, 8])
meal_temperature['hot'] = fuzz.trapmf(meal_temperature.universe, [7, 9, 10, 10])

cleanliness['dirty'] = fuzz.trapmf(cleanliness.universe, [0, 0, 3, 5])
cleanliness['okay'] = fuzz.trimf(cleanliness.universe, [4, 6, 8])
cleanliness['spotless'] = fuzz.trapmf(cleanliness.universe, [7, 9, 10, 10])

customer_satisfaction['dissatisfied'] = fuzz.trapmf(customer_satisfaction.universe, [0, 0, 2, 4])
customer_satisfaction['content'] = fuzz.trimf(customer_satisfaction.universe, [3, 5, 7])
customer_satisfaction['delighted'] = fuzz.trapmf(customer_satisfaction.universe, [6, 8, 10, 10])


rule1 = ctrl.Rule(service_quality['poor'] | meal_temperature['cold'] | cleanliness['dirty'], customer_satisfaction['dissatisfied'])
rule2 = ctrl.Rule(service_quality['average'] & meal_temperature['warm'] & cleanliness['okay'], customer_satisfaction['content'])
rule3 = ctrl.Rule(service_quality['excellent'] & meal_temperature['hot'] & cleanliness['spotless'], customer_satisfaction['delighted'])

satisfaction_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
satisfaction_simulation = ctrl.ControlSystemSimulation(satisfaction_ctrl)

def get_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            if 0 <= value <= 10:
                return value
            else:
                print("Masukkan angka antara 0-10.")
        except ValueError:
            print("Masukkan angka yang valid.")

print("Selamat Datang di Fuzzy Paradise!")
print("Masukkan Nilai Input (0-10):")
service_quality_value = get_input("Kualitas Pelayanan: ")
meal_temperature_value = get_input("Suhu Makanan: ")
cleanliness_value = get_input("Kebersihan: ")

satisfaction_simulation.input['service_quality'] = service_quality_value
satisfaction_simulation.input['meal_temperature'] = meal_temperature_value
satisfaction_simulation.input['cleanliness'] = cleanliness_value

satisfaction_simulation.compute()

output_satisfaction = satisfaction_simulation.output['customer_satisfaction']
if output_satisfaction <= 3:
    satisfaction_level = "Dissatisfied"
elif output_satisfaction <= 7:
    satisfaction_level = "Content"
else:
    satisfaction_level = "Delighted"

print(f"\nTingkat Kepuasan Pelanggan: {output_satisfaction:.2f} (0-10)")
print(f"Kategori: {satisfaction_level}")

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

service_quality.view(ax=axs[0, 0])
axs[0, 0].set_title("Fungsi Keanggotaan: Service Quality")

meal_temperature.view(ax=axs[0, 1])
axs[0, 1].set_title("Fungsi Keanggotaan: Meal Temperature")

cleanliness.view(ax=axs[1, 0])
axs[1, 0].set_title("Fungsi Keanggotaan: Cleanliness")

customer_satisfaction.view(sim=satisfaction_simulation, ax=axs[1, 1])
axs[1, 1].set_title("Hasil Keanggotaan: Customer Satisfaction")

plt.tight_layout()
plt.show()
