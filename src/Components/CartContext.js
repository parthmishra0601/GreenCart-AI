// CartContext.js
import React, { createContext, useContext, useState, useEffect } from "react";

const CartContext = createContext();

export function CartProvider({ children }) {
  const [cartItems, setCartItems] = useState([]);

  // Load cart from local storage
  useEffect(() => {
    const storedCart = localStorage.getItem("greenerCart");
    if (storedCart) {
      setCartItems(JSON.parse(storedCart));
    }
  }, []);

  // Save cart to local storage when it changes
  useEffect(() => {
    localStorage.setItem("greenerCart", JSON.stringify(cartItems));
  }, [cartItems]);

  const addToCart = (product) => {
    const existingItem = cartItems.find(
      (item) => item.greener_alternative === product.greener_alternative
    );

    if (existingItem) {
      setCartItems(
        cartItems.map((item) =>
          item.greener_alternative === product.greener_alternative
            ? { ...item, quantity: item.quantity + 1 }
            : item
        )
      );
    } else {
      setCartItems([...cartItems, { ...product, quantity: 1 }]);
    }
  };

  const removeFromCart = (productName) => {
    setCartItems(
      cartItems.filter((item) => item.greener_alternative !== productName)
    );
  };

  const updateQuantity = (productName, change) => {
    setCartItems(
      cartItems.map((item) =>
        item.greener_alternative === productName
          ? { ...item, quantity: Math.max(1, item.quantity + change) }
          : item
      )
    );
  };

  const clearCart = () => {
    setCartItems([]);
  };

  return (
    <CartContext.Provider
      value={{
        cartItems,
        addToCart,
        removeFromCart,
        updateQuantity,
        clearCart,
      }}
    >
      {children}
    </CartContext.Provider>
  );
}

export function useCart() {
  return useContext(CartContext);
}
