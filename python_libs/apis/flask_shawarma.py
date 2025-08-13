# 🌶️ Flask օրինակ - Երևանյան Շաուրմա

from flask import Flask, request, jsonify
from datetime import datetime
import json

# --- FLASK APP ---
app = Flask(__name__)

# --- "ՏՎՅԱԼՆԵՐԻ ԲԱԶԱ" ---
menu_items = {
    "հավով": {"name": "հավով", "price": 1500, "available": True, "prep_time": 3},
    "տավարով": {"name": "տավարով", "price": 1800, "available": True, "prep_time": 4},
    "բանջարեղենով": {"name": "բանջարեղենով", "price": 1200, "available": True, "prep_time": 2},
    "հատուկ": {"name": "հատուկ", "price": 2200, "available": True, "prep_time": 6}
}

orders_storage = {}
next_order_id = 1

# --- ՕԺԱՆԴԱԿ ՖՈՒՆԿՑԻԱՆԵՐ ---

def validate_menu_items(items):
    """Ստուգել՝ արդյոք պատվիրված ապրանքները կան ցանկում"""
    for item in items:
        if item not in menu_items:
            return False, f"'{item}' շաուրմա մենք չունենք: Առկա տարբերակներ՝ {list(menu_items.keys())}"
        if not menu_items[item]["available"]:
            return False, f"'{item}' շաուրման ժամանակավորապես մատչելի չէ"
    return True, "OK"

def calculate_order_total(items):
    """Հաշվել պատվերի գումարը և պատրաստման ժամանակը"""
    total_price = sum(menu_items[item]["price"] for item in items)
    total_time = max(menu_items[item]["prep_time"] for item in items)
    return total_price, total_time

# --- API ENDPOINTS ---

@app.route("/")
def root():
    """Գլխավոր էջ"""
    return jsonify({
        "message": "Բարի գալուստ Երևանյան Շաուրմա Flask API 🥙",
        "menu_url": "/menu",
        "orders_url": "/orders",
        "version": "1.0.0"
    })

@app.route("/menu", methods=["GET"])
def get_menu():
    """GET /menu - Ցանկի ստացում"""
    return jsonify({
        "status": "success",
        "menu": menu_items
    })

@app.route("/orders", methods=["POST"])
def create_order():
    """POST /orders - Նոր պատվեր ստեղծում"""
    global next_order_id
    
    # Get JSON data
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "JSON data չի գտնվել"}), 400
    
    # Validation
    required_fields = ["customer_name", "items"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Պարտադիր դաշտ '{field}' բացակայում է"}), 400
    
    customer_name = data["customer_name"].strip()
    items = data["items"]
    special_requests = data.get("special_requests", "")
    
    if len(customer_name) < 2:
        return jsonify({"error": "Անունը պետք է լինի նվազագույնը 2 տառ"}), 400
    
    # Validate menu items
    is_valid, error_msg = validate_menu_items(items)
    if not is_valid:
        return jsonify({"error": error_msg}), 404
    
    # Calculate
    total_price, prep_time = calculate_order_total(items)
    
    # Create order
    new_order = {
        "id": next_order_id,
        "customer_name": customer_name,
        "items": items,
        "total_price": total_price,
        "status": "գործընթաց",
        "created_at": datetime.now().isoformat(),
        "estimated_time": prep_time,
        "special_requests": special_requests
    }
    
    orders_storage[next_order_id] = new_order
    order_id = next_order_id
    next_order_id += 1
    
    print(f"🥙 Պատրաստում եմ պատվեր #{order_id} ({customer_name})")
    
    return jsonify({
        "status": "created",
        "order": new_order,
        "message": f"Պատվեր #{order_id} ստեղծվեց: Պատրաստ կլինի {prep_time} րոպեում"
    }), 201

@app.route("/orders/<int:order_id>", methods=["GET"])
def get_order(order_id):
    """GET /orders/{order_id} - Մեկ պատվերի տվյալներ"""
    if order_id not in orders_storage:
        return jsonify({"error": f"Պատվեր #{order_id} չի գտնվել"}), 404
    
    return jsonify({
        "status": "success", 
        "order": orders_storage[order_id]
    })

@app.route("/orders", methods=["GET"])
def get_all_orders():
    """GET /orders - Բոլոր պատվերները"""
    return jsonify({
        "status": "success", 
        "orders": list(orders_storage.values()),
        "total": len(orders_storage)
    })

@app.route("/orders/<int:order_id>", methods=["PUT"])
def update_order(order_id):
    """PUT /orders/{order_id} - Պատվերը փոխել"""
    if order_id not in orders_storage:
        return jsonify({"error": f"Պատվեր #{order_id} չի գտնվել"}), 404
    
    order = orders_storage[order_id]
    if order["status"] != "գործընթաց":
        return jsonify({"error": "Պատրաստ պատվերը չի կարելի փոխել"}), 422
    
    data = request.get_json()
    if not data or "items" not in data:
        return jsonify({"error": "Նոր items դաշտը պարտադիր է"}), 400
    
    new_items = data["items"]
    
    # Validate
    is_valid, error_msg = validate_menu_items(new_items)
    if not is_valid:
        return jsonify({"error": error_msg}), 404
    
    total_price, prep_time = calculate_order_total(new_items)
    
    order["items"] = new_items
    order["total_price"] = total_price
    order["estimated_time"] = prep_time
    
    return jsonify({"status": "updated", "order": order})

@app.route("/orders/<int:order_id>", methods=["DELETE"])
def cancel_order(order_id):
    """DELETE /orders/{order_id} - Պատվերը չեղարկել"""
    if order_id not in orders_storage:
        return jsonify({"error": f"Պատվեր #{order_id} չի գտնվել"}), 404
    
    order = orders_storage[order_id]
    if order["status"] == "պատրաստ":
        return jsonify({"error": "Պատրաստ պատվերը չի կարելի չեղարկել"}), 422
    
    del orders_storage[order_id]
    return jsonify({
        "status": "cancelled", 
        "message": f"Պատվեր #{order_id} չեղարկվեց"
    })

# --- ERROR HANDLERS ---

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Էջը չի գտնվել"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Ներքին սխալ"}), 500

# --- ՄԻՋՈՒԿԱՅԻՆ ԾՐԱԳԻՐ ---
if __name__ == "__main__":
    print("🌶️ Գործարկում եմ Երևանյան Շաուրմա Flask API...")
    print("🌍 Server: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
