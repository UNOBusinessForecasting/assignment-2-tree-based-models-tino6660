from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
y = data['meal']
x = data[['Total','Discounts','Brewed_Coffee_12_oz','Brewed_Coffee_16_oz','Aquafina_Water','Muffin_Pastry_Case','Extra_Syrup','Bottled_Soda_Mt_Dew',
'Bottled_Soda_Diet_Pepsi','Latte_16_oz','Latte_12_oz','Original','String_Cheese','Soda_Fountain_24_oz','Starbucks_DS_Vanilla',
'Bottled_Soda_Wild_Cherry','Gatorade_Glacier_Freeze','TeaSmith_Tea_16_oz','Iced','Bottled_Soda_Diet_Mt_Dew','Candy_Snickers','Tea_Unsweetened',
'Bottled_Soda_Code_Red','Chips_Kettle_Jalapeo','Chex_Mix_Traditional','Rockstar_Zero_Punched','Ocean_Spray_Apple','Americano_16_oz','TeaSmith_Tea_12_oz',
'Tea_Half_and_Half','Chips_Kettle_Sea_Salt','Mocha_16_oz','Nuts_Cashews','Chips_Kettle_BBQ','Tea_Sweetened','Gatorade_Orange','Chex_Mix_Cheddar',
'Candy_KitKat','Parfait_Strawberry','Gatorade_Cool_Blue','Cheez_It_Original','Bottled_Soda_Pepsi','Nuts_Hot_n_Spicy_Peanuts',"Candy_Peanut_Butter_MM's",
'Ocean_Spray_Orange','Kickstart_Raspberry','Kickstart_Orange','Candy_Reeses','Extra_Espresso','Clif_Bar_White_Chocolate_Macadamia_Nut',
'Candy_Twix',"Western's_Jerky_Beef",'Pringles_Original','Nuts_Salted_Peanuts','Pringles_Sour_Cream_Onion','Chocolate_Muscle_Milk',
'Kickstart_Pineapple_Orange_Mango','MM','Turtle_16_oz','Kickstart_Grape','Parfait_Blueberry','Moroccan_Mint','SoBe_Life_Acai','White_Mocha_16_oz',
'Gum_Spearmint','Kickstart_Mango_Lime','Clif_Bar_Brownie_Bar','Cheez_It_White_Cheddar','Starbucks_Mocha_Frap','Gatorade_Fruit_Punch',
'Hot_Chocolate_12_oz','Cup','Sweet_Pomegranate','Gatorade_Grape','White_Mocha_12_oz','Whipped_Topping','Starbucks_DS_Mocha','Hot_Chocolate_16_oz',
'Dr_Pepper','Ocean_Spray_CranGrape','Gum_Peppermint']]

x, xt, y, yt = train_test_split(x, y, test_size=0.3)

model = RF(n_estimators=100, n_jobs=-1, max_depth=5)
modelFit = model.fit(x,y)

data2 = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")
X_2 = data2.drop(columns=['meal'])
pred = model.predict(X_2)
