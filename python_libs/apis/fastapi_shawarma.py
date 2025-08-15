# 🚀 Ամբողջական FastAPI օրինակ - Երևանյան Շաուրմա

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import Optional, List
from datetime import datetime
import asyncio

# --- FASTAPI APP ---
app = FastAPI(
    title="Yerevanyan Shawarma API",
    description="🥙 Երևանյան Շաուրմա - API for ordering delicious shawarma",
    version="1.0.0",
    docs_url="/docs", # default
    redoc_url="/redoc" # default
)

# --- ՏՎՅԱԼՆԵՐԻ ՄՈԴԵԼՆԵՐ ---
class ShawarmaItem(BaseModel):
    name: str
    price: int
    available: bool = True
    prep_time: int  # րոպեներ

class OrderCreate(BaseModel):
    customer_name: str
    items: List[str]
    special_requests: Optional[str] = ""
    
    @field_validator('customer_name')
    @classmethod
    def validate_name(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('Անունը պետք է լինի նվազագույնը 2 տառ')
        return v.strip()

class Order(BaseModel):
    id: int
    customer_name: str
    items: List[str]
    total_price: int
    status: str
    created_at: datetime
    estimated_time: int

class OrderResponse(BaseModel):
    status: str
    order: Order
    message: str

# --- "ՏՎՅԱԼՆԵՐԻ ԲԱԶԱ" ---
menu_items = {
    "հավով": ShawarmaItem(name="հավով", price=1500, prep_time=3),
    "տավարով": ShawarmaItem(name="տավարով", price=1800, prep_time=4),
    "բանջարեղենով": ShawarmaItem(name="բանջարեղենով", price=1200, prep_time=2),
    "հատուկ": ShawarmaItem(name="հատուկ", price=2200, prep_time=6)
}

orders_storage = {}
next_order_id = 1

# --- ՕԺԱՆԴԱԿ ՖՈՒՆԿՑԻԱՆԵՐ ---

async def validate_menu_items(items: List[str]) -> None:
    """Ստուգել՝ արդյոք պատվիրված ապրանքները կան ցանկում"""
    for item in items:
        if item not in menu_items:
            raise HTTPException(
                status_code=404, 
                detail=f"'{item}' շաուրմա մենք չունենք: Առկա տարբերակներ՝ {list(menu_items.keys())}"
            )
        if not menu_items[item].available:
            raise HTTPException(
                status_code=422,
                detail=f"'{item}' շաուրման ժամանակավորապես մատչելի չէ"
            )

def calculate_order_total(items: List[str]) -> tuple[int, int]:
    """Հաշվել պատվերի գումարը և պատրաստման ժամանակը"""
    total_price = sum(menu_items[item].price for item in items)
    total_time = max(menu_items[item].prep_time for item in items)  # Ամենաերկար ժամանակ
    return total_price, total_time

# --- API ENDPOINTS ---

@app.get("/")
async def root():
    """Գլխավոր էջ"""
    return {
        "message": "Բարի գալուստ Երևանյան Շաուրմա API 🥙",
        "docs": "/docs",
        "version": "1.0.0"
    }

@app.get("/menu")
async def get_menu():
    """GET /menu - Ցանկի ստացում"""
    return {
        "status": "success",
        "menu": {name: {"price": item.price, "available": item.available, "prep_time": item.prep_time} 
                for name, item in menu_items.items()}
    }

@app.post("/orders", response_model=OrderResponse)
async def create_order(order_data: OrderCreate):
    """POST /orders - Նոր պատվեր ստեղծում"""
    global next_order_id
    
    # Validation
    await validate_menu_items(order_data.items)
    
    # Calculation
    total_price, prep_time = calculate_order_total(order_data.items)
    
    # Create order
    new_order = Order(
        id=next_order_id,
        customer_name=order_data.customer_name,
        items=order_data.items,
        total_price=total_price,
        status="գործընթաց",
        created_at=datetime.now(),
        estimated_time=prep_time
    )
    
    orders_storage[next_order_id] = new_order
    order_id = next_order_id
    next_order_id += 1
    
    # Simulate cooking
    print(f"🥙 Պատրաստում եմ պատվեր #{order_id} ({order_data.customer_name})")
    await asyncio.sleep(0.5)  # Simulation
    
    return OrderResponse(
        status="created",
        order=new_order,
        message=f"Պատվեր #{order_id} ստեղծվեց: Պատրաստ կլինի {prep_time} րոպեում"
    )

@app.get("/orders/{order_id}")
async def get_order(order_id: int):
    """GET /orders/{order_id} - Մեկ պատվերի տվյալներ"""
    if order_id not in orders_storage:
        raise HTTPException(status_code=404, detail=f"Պատվեր #{order_id} չի գտնվել")
    
    return {"status": "success", "order": orders_storage[order_id]}

@app.get("/orders")
async def get_all_orders():
    """GET /orders - Բոլոր պատվերները"""
    return {
        "status": "success", 
        "orders": list(orders_storage.values()),
        "total": len(orders_storage)
    }

@app.put("/orders/{order_id}")
async def update_order(order_id: int, new_items: List[str]):
    """PUT /orders/{order_id} - Պատվերը փոխել"""
    if order_id not in orders_storage:
        raise HTTPException(status_code=404, detail=f"Պատվեր #{order_id} չի գտնվել")
    
    order = orders_storage[order_id]
    if order.status != "գործընթաց":
        raise HTTPException(status_code=422, detail="Պատրաստ պատվերը չի կարելի փոխել")
    
    await validate_menu_items(new_items)
    total_price, prep_time = calculate_order_total(new_items)
    
    order.items = new_items
    order.total_price = total_price
    order.estimated_time = prep_time
    
    return {"status": "updated", "order": order}

@app.delete("/orders/{order_id}")
async def cancel_order(order_id: int):
    """DELETE /orders/{order_id} - Պատվերը չեղարկել"""
    if order_id not in orders_storage:
        raise HTTPException(status_code=404, detail=f"Պատվեր #{order_id} չի գտնվել")
    
    order = orders_storage[order_id]
    if order.status == "պատրաստ":
        raise HTTPException(status_code=422, detail="Պատրաստ պատվերը չի կարելի չեղարկել")
    
    del orders_storage[order_id]
    return {"status": "cancelled", "message": f"Պատվեր #{order_id} չեղարկվեց"}

# --- ՄԻՋՈՒԿԱՅԻՆ ԾՐԱԳԻՐ ---
if __name__ == "__main__":
    import uvicorn
    print("🚀 Գործարկում եմ Երևանյան Շաուրմա API...")
    print("📖 Docs: http://localhost:8000/docs")
    print("🔄 ReDoc: http://localhost:8000/redoc")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
