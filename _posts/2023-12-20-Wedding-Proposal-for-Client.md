---
title: "Wedding Proposal for Client"
date: 2023-12-20
tags: [SQL, Python]
header:
  image: "/images/wedding_proposal_banner.jpg"
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
    
Choose our services for your wedding and entrust the planning to the very best hands, leaving you absolutely relaxed to enjoy the best day of your lives!<br><br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<i>-The Matrix Bliss Events</i>


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
### Wedding Proposal
Wedding Size: Medium (51 - 150)<br>
Budget Level: Moderate (135 guests)

<img src="{{ site.url }}{{ site.baseurl }}/images/w_proposal_01.png" alt="">
<img src="{{ site.url }}{{ site.baseurl }}/images/w_proposal_02.png" alt="">
<img src="{{ site.url }}{{ site.baseurl }}/images/w_proposal_03.png" alt="">
