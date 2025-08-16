from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel, field_validator, Field
from typing import Optional, List
from datetime import datetime
import asyncio
import uvicorn
from enum import Enum
import time


app = FastAPI(
    title="Yerevanyan Shawarma API",
    description="🥙 Երևանյան Շաուրմա - API for ordering delicious shawarma",
    version="1.0.0",
    docs_url="/docs", # default
    redoc_url="/redoc" # default
)

class ShawarmaType(Enum):
    HAVOV = "հավով"
    TAVAROV = "տավարով"
    BANJAR = "բանջարեղենով"
    HATUK = "հատուկ"

class ShawarmaItem(BaseModel):
    name: ShawarmaType
    price: int
    available: bool = True
    prep_time: int  # րոպեներ

class OrderCreate(BaseModel):
    customer_name: str
    items: List[ShawarmaType]
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
    items: List[ShawarmaType]
    # total_price: int
    status: str
    created_at: datetime
    # estimated_time: int

class OrderResponse(BaseModel):
    status: str
    order: Order
    message: str
    

menu_items = {
    ShawarmaType.HAVOV: ShawarmaItem(name=ShawarmaType.HAVOV, price=1500, prep_time=3),
    ShawarmaType.TAVAROV: ShawarmaItem(name=ShawarmaType.TAVAROV, price=1800, prep_time=4),
    ShawarmaType.BANJAR: ShawarmaItem(name=ShawarmaType.BANJAR, price=1200, prep_time=2),
    ShawarmaType.HATUK: ShawarmaItem(name=ShawarmaType.HATUK, price=2200, prep_time=6)
}

orders_storage = {}
next_order_id = 1

@app.get("/ping")
def ping(): # health
    return {
        "message": "Բարի գալուստ Երևանյան Շաուրմա API 🥙",
        "docs": "/docs",
        "version": "1.0.0"
    }

@app.get("/menu")
def get_menu():
    return {
        "status": "success",
        "menu": {name: {"price": item.price, "available": item.available, "prep_time": item.prep_time} for name, item in menu_items.items()}
    }

# get concrete item from menu
@app.get("/menu/{item_name}")
def get_menu_item(item_name: str):
    
    if item_name not in menu_items:
        raise HTTPException(status_code=404, 
                            detail=f"{item_name} not found. Options are: {list(menu_items.keys())}")
    item = menu_items[item_name]
    
    return {
        "status": "success",
        "menu_item": {
            "name": item.name,
            "price": item.price,
            "available": item.available,
            "prep_time": item.prep_time
        }
    }
    
@app.post("/orders", response_model=OrderResponse)
def create_order(order_data: OrderCreate):
    """Create a new order.

    Args:
        order_data (OrderCreate): The order data to create a new order.

    Returns:
        _type_: _description_
    """
    global next_order_id
    
    new_order = Order(
        id=next_order_id,
        customer_name=order_data.customer_name,
        items=order_data.items,
        # total_price=total_price,
        status="Ընթացքում ա",
        created_at=datetime.now(),
        # estimated_time=prep_time
    )

    orders_storage[next_order_id] = new_order
    next_order_id += 1

    print(f"🥙 Պատրաստում եմ պատվեր #{next_order_id - 1} ({order_data.customer_name})")
    time.sleep(1)

    return OrderResponse(
        status="Պատրաստ ա",
        order = new_order, 
        message=f"Պատվեր #{next_order_id - 1} ստեղծվեց:"
    )
    
    
@app.get("/orders/{order_id}")
async def get_order(order_id: int):
    """GET /orders/{order_id} - Մեկ պատվերի տվյալներ"""
    if order_id not in orders_storage:
        raise HTTPException(status_code=404, detail=f"Պատվեր #{order_id} չի գտնվել")
    
    return {"status": "success", "order": orders_storage[order_id]}

@app.put("/orders/{order_id}")
async def update_order(order_id: int, new_items: List[str]):
    """PUT /orders/{order_id} - Պատվերը փոխել"""
    if order_id not in orders_storage:
        raise HTTPException(status_code=404, detail=f"Պատվեր #{order_id} չի գտնվել")
    
    order = orders_storage[order_id]
    if order.status != "Ընթացքում ա":
        raise HTTPException(status_code=422, detail="Պատրաստ պատվերը չի կարելի փոխել")
    
    # await validate_menu_items(new_items)
    # total_price, prep_time = calculate_order_total(new_items)
    
    order.items = new_items
    # order.total_price = total_price
    # order.estimated_time = prep_time

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


if __name__ == "__main__":
    print("🚀 Գործարկում եմ Երևանյան Շաուրմա API...")
    uvicorn.run(app, port=8000)
