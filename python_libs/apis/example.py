#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🥙 Երևանյան Շաուրմա API - Ամենապարզ օրինակներ
==========================================

Այս ֆայլը ցույց է տալիս FastAPI Shawarma API-ի օգտագործումը պարզ կոդով:

🚀 Նախապես գործարկեք FastAPI սերվերը:
   python fastapi_shawarma.py

📋 Endpoints-ներ:
- GET /               - Գլխավոր էջ
- GET /menu          - Ցանկի ստացում  
- POST /orders       - Նոր պատվեր ստեղծում
- GET /orders/{id}   - Մեկ պատվերի տվյալներ
- GET /orders        - Բոլոր պատվերները
- PUT /orders/{id}   - Պատվերը փոխել
- DELETE /orders/{id} - Պատվերը չեղարկել
"""

import asyncio
import aiohttp
import json

# === ՕՐԻՆԱԿ 1: GET / - Գլխավոր էջ ===
async def example_1_root():
    print("🔸 ՕՐԻՆԱԿ 1: Գլխավոր էջ")
    print("=" * 40)
    
    async with aiohttp.ClientSession() as session:
        async with session.get('http://localhost:8000/') as response:
            result = await response.json()
            print(f"Status: {response.status}")
            print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")

# === ՕՐԻՆԱԿ 2: GET /menu - Մենու ===
async def example_2_menu():
    print("\n🔸 ՕՐԻՆԱԿ 2: Մենու")
    print("=" * 40)
    
    async with aiohttp.ClientSession() as session:
        async with session.get('http://localhost:8000/menu') as response:
            result = await response.json()
            print(f"Status: {response.status}")
            print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")

# === ՕՐԻՆԱԿ 3: POST /orders - Նոր պատվեր ===
async def example_3_create_order():
    print("\n🔸 ՕՐԻՆԱԿ 3: Նոր պատվեր ստեղծել")
    print("=" * 40)
    
    order_data = {
        "customer_name": "Արամ",
        "items": ["հավով", "տավարով"],
        "special_requests": "Խնդրում եմ շատ սոխ"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post('http://localhost:8000/orders', json=order_data) as response:
            result = await response.json()
            print(f"Status: {response.status}")
            print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            # Պատվերի ID-ն վերադարձնենք հետագա օգտագործման համար
            if 'order' in result and 'id' in result['order']:
                return result['order']['id']
    return None

# === ՕՐԻՆԱԿ 4: GET /orders/{id} - Կոնկրետ պատվեր ===
async def example_4_get_order(order_id):
    print(f"\n🔸 ՕՐԻՆԱԿ 4: Պատվեր #{order_id} ստուգել")
    print("=" * 40)
    
    async with aiohttp.ClientSession() as session:
        async with session.get(f'http://localhost:8000/orders/{order_id}') as response:
            result = await response.json()
            print(f"Status: {response.status}")
            print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")

# === ՕՐԻՆԱԿ 5: GET /orders - Բոլոր պատվերները ===
async def example_5_all_orders():
    print("\n🔸 ՕՐԻՆԱԿ 5: Բոլոր պատվերները")
    print("=" * 40)
    
    async with aiohttp.ClientSession() as session:
        async with session.get('http://localhost:8000/orders') as response:
            result = await response.json()
            print(f"Status: {response.status}")
            print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")

# === ՕՐԻՆԱԿ 6: PUT /orders/{id} - Պատվերը փոխել ===
async def example_6_update_order(order_id):
    print(f"\n🔸 ՕՐԻՆԱԿ 6: Պատվեր #{order_id} փոխել")
    print("=" * 40)
    
    new_items = ["հատուկ"]  # Փոխենք հատուկ շաուրմայի
    
    async with aiohttp.ClientSession() as session:
        async with session.put(f'http://localhost:8000/orders/{order_id}', json=new_items) as response:
            result = await response.json()
            print(f"Status: {response.status}")
            print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")

# === ՕՐԻՆԱԿ 7: DELETE /orders/{id} - Պատվերը չեղարկել ===
async def example_7_delete_order(order_id):
    print(f"\n🔸 ՕՐԻՆԱԿ 7: Պատվեր #{order_id} չեղարկել")
    print("=" * 40)
    
    async with aiohttp.ClientSession() as session:
        async with session.delete(f'http://localhost:8000/orders/{order_id}') as response:
            result = await response.json()
            print(f"Status: {response.status}")
            print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")

# === ԳԼԽԱՎՈՐ ՑՈՒՑԱԴՐՈՒԹՅՈՒՆ ===
async def main():
    print("🥙 ԵՐԵՎԱՆՅԱՆ ՇԱՈՒՐՄԱ API - ՊԱՐԶ ՕՐԻՆԱԿՆԵՐ")
    print("=" * 60)
    
    try:
        # 1. Գլխավոր էջ
        await example_1_root()
        await asyncio.sleep(1)
        
        # 2. Մենու
        await example_2_menu()
        await asyncio.sleep(1)
        
        # 3. Նոր պատվեր
        order_id = await example_3_create_order()
        await asyncio.sleep(1)
        
        # 4. Կոնկրետ պատվեր (եթե ստեղծվեց)
        if order_id:
            await example_4_get_order(order_id)
            await asyncio.sleep(1)
            
            # 5. Բոլոր պատվերները
            await example_5_all_orders()
            await asyncio.sleep(1)
            
            # 6. Պատվերը փոխել
            await example_6_update_order(order_id)
            await asyncio.sleep(1)
            
            # 7. Պատվերը չեղարկել
            await example_7_delete_order(order_id)
        
        print("\n✅ Բոլոր օրինակները ավարտվեցին!")
        
    except aiohttp.ClientConnectorError:
        print("\n❌ Սերվերը չի գործում!")
        print("💡 Խնդրում ենք նախ գործարկել: python fastapi_shawarma.py")
    except Exception as e:
        print(f"\n❌ Սխալ: {e}")

# === CURL ՕՐԻՆԱԿՆԵՐ ===
def show_curl_examples():
    print("📋 CURL ՕՐԻՆԱԿՆԵՐ")
    print("=" * 50)
    
    examples = [
        ("1. Գլխավոր էջ", "curl http://localhost:8000/"),
        ("2. Մենու", "curl http://localhost:8000/menu"),
        ("3. Նոր պատվեր", 'curl -X POST "http://localhost:8000/orders" -H "Content-Type: application/json" -d \'{"customer_name": "Արամ", "items": ["հավով"]}\''),
        ("4. Բոլոր պատվերները", "curl http://localhost:8000/orders"),
        ("5. Կոնկրետ պատվեր", "curl http://localhost:8000/orders/1"),
        ("6. Պատվերը փոխել", 'curl -X PUT "http://localhost:8000/orders/1" -H "Content-Type: application/json" -d \'["տավարով"]\''),
        ("7. Պատվերը չեղարկել", "curl -X DELETE http://localhost:8000/orders/1")
    ]
    
    for title, command in examples:
        print(f"\n🔸 {title}")
        print(f"   {command}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "curl":
        show_curl_examples()
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n\n⏹️  Ցուցադրությունը դադարեցվեց")
        except Exception as e:
            print(f"\n❌ Սխալ: {e}")



