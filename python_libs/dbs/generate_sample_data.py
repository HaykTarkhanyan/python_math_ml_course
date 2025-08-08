"""
Armenian Cultural E-commerce Database Generator
==============================================

This script generates authentic Armenian-themed sample data for an e-commerce database with:
- Armenian customers with traditional names from across Armenia and Artsakh
- Orders using Armenian payment systems (ARCA, ACBA, Idram, TelCell)
- Products featuring exclusively Armenian cultural items:
  * Traditional musical instruments (duduk, zurna, oud, kanun, dhol)
  * Handcrafted items (carpets, pottery, khachkars)
  * Armenian wines, spirits, and delicacies
  * Armenian literature and cultural books
  * Traditional games and clothing

The data celebrates Armenian heritage and can be used for database learning with cultural context.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_data(output_dir='dbs', seed=42):
    """
    Generate Armenian-themed e-commerce sample data and save as CSV files.
    
    Features Armenian names, cities, traditional products, and local payment methods.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save CSV files
    seed : int
        Random seed for reproducible results
    
    Returns:
    --------
    tuple
        (customers_df, orders_df, products_df)
    """
    
    # Set random seed for reproducible results
    np.random.seed(seed)
    
    print("🔄 Generating Armenian-themed e-commerce data...")
    
    # === CUSTOMERS DATA ===
    customers = pd.DataFrame({
        'customer_id': range(1, 21),
        'name': [
            'Armen Sargsyan', 'Anahit Hakobyan', 'Davit Grigoryan', 'Sona Petrosyan', 'Vahram Manukyan',
            'Karine Abrahamyan', 'Tigran Avetisyan', 'Lilit Gevorgyan', 'Hayk Khachatryan', 'Lusine Vardanyan',
            'Aram Hovhannisyan', 'Nvard Karapetyan', 'Ruben Danielyan', 'Arusyak Mkrtchyan', 'Gagik Stepanyan',
            'Siranush Badalyan', 'Arman Ghukasyan', 'Astghik Torosyan', 'Vazgen Poghosyan', 'Gayane Mesropyan'
        ],
        'email': [
            'armen.s@armmail.am', 'anahit.h@haypost.am', 'davit.g@arca.am', 'sona.p@armmail.am', 'vahram.m@ucom.am',
            'karine.a@haypost.am', 'tigran.av@gmail.com', 'lilit.g@armmail.am', 'hayk.kh@ucom.am', 'lusine.v@arca.am',
            'aram.h@gmail.com', 'nvard.k@armmail.am', 'ruben.d@haypost.am', 'arusyak.m@ucom.am', 'gagik.s@arca.am',
            'siranush.b@gmail.com', 'arman.gh@armmail.am', 'astghik.t@haypost.am', 'vazgen.p@ucom.am', 'gayane.m@gmail.com'
        ],
        'country': [
            'Armenia', 'Armenia', 'Armenia', 'Armenia', 'Armenia', 'Armenia', 'Armenia', 'Armenia', 'Armenia', 'Armenia',
            'Armenia', 'Armenia', 'Artsakh', 'Armenia', 'Armenia', 'Armenia', 'Artsakh', 'Armenia', 'Armenia', 'Armenia'
        ],
        'city': [
            'Yerevan', 'Gyumri', 'Vanadzor', 'Yerevan', 'Kapan', 'Yerevan', 'Goris', 'Sisian', 'Yerevan', 'Alaverdi',
            'Artashat', 'Charentsavan', 'Stepanakert', 'Hrazdan', 'Yerevan', 'Dilijan', 'Shushi', 'Yerevan', 'Sevan', 'Abovyan'
        ],
        'age': np.random.randint(18, 65, 20),
        'registration_date': pd.date_range(start='2023-01-01', end='2024-12-31', periods=20),
        'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], 20, p=[0.2, 0.5, 0.3])
    })
    
    # === PRODUCTS DATA ===
    products = pd.DataFrame({
        'product_id': range(1, 26),
        'product_name': [
            'Handcrafted Armenian Duduk', 'Professional Zurna', 'Traditional Oud', 'Armenian Kanun', 'Dhol Drum Set',
            'Handwoven Karabagh Carpet', 'Areni-1 Red Wine', 'Armenian Cognac Ararat 20yr', 'Dried Armenian Apricots', 'Fresh Lavash Bread',
            'Armenian History by Hovhannes Tumanyan', 'Komitas Sacred Music Collection', 'Western Armenian Grammar Book', 'Artsakh Heritage Album', 'Traditional Armenian Recipes',
            'Handmade Ceramic Khachkar', 'Armenian Coffee Service Set', 'Clay Tonir for Baking', 'Wild Armenian Herbs Bundle', 'Artsakh Pomegranate Wine',
            'Carved Wooden Chess Set', 'Traditional Nardi (Backgammon)', 'Mount Ararat Climbing Equipment', 'Embroidered Traditional Taraz', 'Armenian Wrestling Singlet'
        ],
        'category': [
            'Musical Instruments', 'Musical Instruments', 'Musical Instruments', 'Musical Instruments', 'Musical Instruments',
            'Traditional Crafts', 'Armenian Wines', 'Armenian Spirits', 'Food & Delicacies', 'Food & Delicacies',
            'Armenian Literature', 'Armenian Literature', 'Armenian Literature', 'Armenian Literature', 'Armenian Literature',
            'Traditional Crafts', 'Traditional Crafts', 'Traditional Crafts', 'Food & Delicacies', 'Armenian Wines',
            'Traditional Games', 'Traditional Games', 'Sports Equipment', 'Traditional Clothing', 'Sports Equipment'
        ],
        'price': [450.00, 280.00, 350.00, 750.00, 320.00,
                  1200.00, 85.00, 250.00, 25.99, 8.50,
                  39.99, 29.99, 35.99, 49.99, 42.99,
                  150.00, 89.99, 450.00, 18.99, 65.00,
                  120.00, 180.00, 299.99, 180.00, 75.00],
        'stock_quantity': np.random.randint(5, 50, 25),
        'supplier': np.random.choice(['YerevanCrafts', 'AraratWines', 'TraditionalArts', 'ArmenianHeritage', 'GyumriWorkshop'], 25)
    })
    
    # === ORDERS DATA ===
    order_data = []
    order_id = 101
    base_date = datetime(2024, 1, 1)
    
    # Create orders for customers with varying patterns
    for customer_id in customers['customer_id']:
        # Each customer gets 1-8 orders
        num_orders = np.random.randint(1, 9)
        customer_segment = customers[customers['customer_id'] == customer_id]['customer_segment'].iloc[0]
        
        for _ in range(num_orders):
            # Order date spread across 2024
            days_offset = np.random.randint(0, 365)
            order_date = base_date + timedelta(days=days_offset)
            
            # Amount varies by customer segment
            if customer_segment == 'Premium':
                amount = np.random.uniform(150, 500)
            elif customer_segment == 'Standard':
                amount = np.random.uniform(50, 200)
            else:  # Basic
                amount = np.random.uniform(15, 100)
            
            # Product categories
            product_category = np.random.choice([
                'Musical Instruments', 'Traditional Crafts', 'Armenian Literature', 'Food & Delicacies', 'Traditional Games', 
                'Armenian Wines', 'Armenian Spirits', 'Traditional Clothing', 'Sports Equipment'
            ])
            
            # Payment methods
            payment_method = np.random.choice(['ARCA Card', 'ACBA Bank Transfer', 'Cash', 'Idram', 'TelCell'], p=[0.3, 0.25, 0.2, 0.15, 0.1])
            
            # Shipping methods
            shipping_method = np.random.choice(['HayPost Standard', 'Yerevan Express', 'Same Day Delivery'], p=[0.6, 0.3, 0.1])
            
            # Order status
            order_status = np.random.choice(['Completed', 'Processing', 'Shipped', 'Cancelled'], p=[0.7, 0.1, 0.15, 0.05])
            
            order_data.append({
                'order_id': order_id,
                'customer_id': customer_id,
                'amount': round(amount, 2),
                'order_date': order_date,
                'product_category': product_category,
                'payment_method': payment_method,
                'shipping_method': shipping_method,
                'order_status': order_status,
                'discount_applied': np.random.choice([0, 5, 10, 15, 20], p=[0.4, 0.3, 0.15, 0.1, 0.05])
            })
            order_id += 1
    
    orders = pd.DataFrame(order_data)
    
    # === SAVE TO CSV FILES ===
    os.makedirs(output_dir, exist_ok=True)
    
    customers_file = os.path.join(output_dir, 'customers.csv')
    orders_file = os.path.join(output_dir, 'orders.csv')
    products_file = os.path.join(output_dir, 'products.csv')
    
    customers.to_csv(customers_file, index=False)
    orders.to_csv(orders_file, index=False)
    products.to_csv(products_file, index=False)
    
    print(f"✅ Data generated and saved:")
    print(f"   📄 {customers_file} - {len(customers)} customers")
    print(f"   📄 {orders_file} - {len(orders)} orders")
    print(f"   📄 {products_file} - {len(products)} products")
    print(f"   📅 Date range: {orders['order_date'].min().strftime('%Y-%m-%d')} to {orders['order_date'].max().strftime('%Y-%m-%d')}")
    print(f"   💰 Total revenue: ${orders['amount'].sum():,.2f}")
    print(f"   📊 Average order value: ${orders['amount'].mean():.2f}")
    
    return customers, orders, products

def load_data_to_sqlite(csv_dir='dbs', db_path=':memory:'):
    """
    Load CSV data into SQLite database.
    
    Parameters:
    -----------
    csv_dir : str
        Directory containing CSV files
    db_path : str
        SQLite database path (':memory:' for in-memory database)
    
    Returns:
    --------
    sqlite3.Connection
        Database connection object
    """
    import sqlite3
    
    print(f"🔄 Loading data into SQLite database...")
    
    # Read CSV files
    customers = pd.read_csv(os.path.join(csv_dir, 'customers.csv'))
    orders = pd.read_csv(os.path.join(csv_dir, 'orders.csv'))
    products = pd.read_csv(os.path.join(csv_dir, 'products.csv'))
    
    # Convert date columns
    customers['registration_date'] = pd.to_datetime(customers['registration_date'])
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    
    # Create SQLite connection
    conn = sqlite3.connect(db_path)
    
    # Load data into SQLite
    customers.to_sql('customers', conn, index=False, if_exists='replace')
    orders.to_sql('orders', conn, index=False, if_exists='replace')
    products.to_sql('products', conn, index=False, if_exists='replace')
    
    print(f"✅ Data loaded into SQLite:")
    print(f"   🗃️ Database: {db_path}")
    print(f"   📊 Tables: customers, orders, products")
    
    return conn

if __name__ == "__main__":
    # Generate sample data
    customers, orders, products = generate_sample_data()
    
    # Load into SQLite and show sample queries
    conn = load_data_to_sqlite()
    
    print("\n📊 Sample Query Results:")
    print("\nCustomers by country:")
    result = pd.read_sql_query("""
        SELECT country, COUNT(*) as customer_count
        FROM customers 
        GROUP BY country 
        ORDER BY customer_count DESC
    """, conn)
    print(result)
    
    print("\nTop 5 orders by amount:")
    result = pd.read_sql_query("""
        SELECT c.name, o.amount, o.product_category, o.order_date
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        ORDER BY o.amount DESC
        LIMIT 5
    """, conn)
    print(result)
    
    conn.close()
