---
title: "Wedding Proposal for Client"
date: 2023-12-20
tags: [SQL, Python]
header:
  image: "/images/hmua.jpg"
excerpt: "(SQL) Designed a database based on the hair and makeup vendors information in San Francisco Bay Area. This includes entities, functional dependencies, and Entity Relationship Diagram in 3NF."
mathjax: "true"
toc: true
toc_label : "Navigate":
---
By: Team Matrix <br>
Michelle Kae Celine Jo-anne Bantugon (Data Engineer) <br>
Miron Tkachuk (Data Engineer) <br>
Srinivas Kondeti (Product Manager) <br>
Zamambo Mkhize (Business-Side Stakeholder) <br>

Business Challenge built by [Professor Chase Kusterer](https://github.com/chase-kusterer)<br>
Hult International Business School<br>



### Dear Bride & Groom,

<em>&emsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Congratulations on your engagement!<br>
    &emsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ready to start planning the best day of your lives?</em><br>
    
Choose our services for your wedding and entrust the planning to the very best hands, leaving you absolutely relaxed to enjoy the best day of your lives!<br>

&emsp;&emsp;&emsp;&nbsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - The Matrix Bliss Events

<div class="alert alert-block alert-success">
Proving  the elegance of simplicity in color palettes, this ethereal styled wedding presentation showcases greenery-filled floral arrangements. It offers exquisite styling inspiration for a romantic-vintage wedding, all presented in the heavenly hues of dusty blue.
</div>

<div class="alert alert-block alert-info">
    <b>Wedding Theme:</b> Romantic + Vintage &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<b>Setting:</b>&emsp;&emsp;Indoor + Outdoor  <br>
    <b>Color:</b>&emsp;&emsp;&emsp;&emsp;&emsp; Dusty Blue &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<b>Location:</b>&emsp;601 Murray Cir, Sausalito, CA 94965
</div>
<br>

<div style="display: flex; justify-content: center; align-items: center;">
<img src="theme.jpg" alt="theme.jpg" width="300" height="200" style="float:left; margin-right: -200px;" />
<img src="invitation.png" alt="invitation.png" width="200" height="200" style="margin: 0 240px;" />
</div>
</div>
</div>

<div class="alert alert-block alert-warning" style="text-align: center; font-family: 'Calibri Light', sans-serif; font-size: 18px;">
VENDORS<br>
<hr>  
<span style='font-family: "Arial"; font-size: 14px; color: gray; display: block; text-align: left;'>&emsp;&emsp;VENUE: Cavallo Point The Lodge at the Golden Gate&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;HAIR & MAKEUP: Shineforth Salon and The Pretty Committee &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<br><br>
&emsp;&emsp;RENTALS: Eventlyst &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;DRESS & ATTIRE: Stacees (Bride Dress), Blacktux (Tie), Misaac (B.maid Dress) &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; 
<br>&emsp;&emsp;CATERING: Fogcutter &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;JEWELRY: Diamondere <br><br>
&emsp;&emsp;FLOWERS: Expressions Floral &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; PHOTO AND VIDEO: Danny Rey Films<br> <br>
&emsp;&emsp;INVITATION:The Knot & Paper culture &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;MUSIC: iMobile DJs <br> <br>
<span>    
</div> 
</div>

### Code
```
# Importing pandas libraries
import pandas as pd

# Storing the path to the dataset
# Change the path to './wedding_cost_data.xlsx' before the submit
path = './wedding_cost_data.xlsx'

# Instantiating the dataset as an object
wedding_data = pd.read_excel(path)

# Displaying the first 5 rows of the dataset
wedding_data.head(n=5)
```

```
# Defining a function to break down cost by department
def department_breakdown(data, department):

    # Filtering the dataset to include only the specified department
    filter = (data['category'] == department)
    depart_data = data[filter]
    
    print(f"""From our {department} department we prepared {depart_data.shape[0]} unique product/s!
    """)
    
    # Displaying a breakdown of each product, including vendor, quantity, price, and total cost
    for number,product in depart_data.iterrows():
        
        print(f"""{number+1}. {product['item_name']} provided by {product['vendor_name']}
                                                                                        {product['quantity']} x {round(product['price_per_item'],2)}$ = {product['subtotal']}$""")

    print(f"""
                                                                                        Total:      {round(depart_data['subtotal'].sum(),2)}$""")
    print('------------------------------------------------------------------------------------------------------------')
```

```
# Defining a function to calculate and display the total cost for a specific department
def department_total(data, department):

    # Filtering the dataset to include only the specified department
    filter = (data['category'] == department)
    depart_data = data[filter]
    
    # Displaying the total cost for the specified department.
    print(f"""{department}  total: 
                                  {round(depart_data['subtotal'].sum(),2)}$""")
```

```
# Importing package for colored text and background
from IPython.display import HTML, display

# Defining a function to print colored text with specified background color, text color, and font size
def print_colored(text, background_color, text_color, font_size="16px"):
    style = f"background-color: rgb({background_color}); color: {text_color}; padding: 8px; font-weight: bold; font-size: {font_size}; text-align: center;"
    html_text = f"<div style='{style}'>{text}</div>"
    display(HTML(html_text))

# Displaying a header with colored text, background, and specified font size
print_colored("VENDORS & PRICING", background_color="120,190,210", text_color="white", font_size="40px")

departments = wedding_data['category'].unique()
print(f"""
Guest:  135
Budget: 40000$
          """)

# Iterating through unique departments and displaying breakdown and total
for department in departments:
    department_breakdown(wedding_data, department)

for department in departments:
    department_total(wedding_data, department)


print("******************************************")
# Displaying the overall total cost for the wedding with colored background and bold text
total_cost = str(wedding_data['subtotal'].sum())
formatted_total_cost = '\x1b[1m\x1b[41;184;219;46m' + f'Wedding Cost:                    {round(float(total_cost), 2)}$' + '\x1b[0m\x1b[0m'  # Adding ANSI escape codes for bold and color
print(formatted_total_cost)
```

