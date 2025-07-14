// Cart.jsx
import React from "react"; // Removed useState as cartItems will come from context
import { useNavigate } from "react-router-dom";
import { FaTrash } from "react-icons/fa";
import { useCart } from "./CartContext";

export default function Cart() {
  const navigate = useNavigate();
  // Destructure cartItems and removeFromCart from the context
  const { cartItems, removeFromCart } = useCart();

  // Calculate total from the cartItems obtained from context
  const total = cartItems.reduce((sum, item) => sum + item.price, 0);

  const handleStartShopping = () => {
    navigate("/"); // Function to redirect to the home page
  };

  return (
    <div className="flex min-h-screen w-full bg-gray-50 font-sans overflow-hidden">
      <main className="flex-grow p-8 overflow-y-auto">
        <h2 className="text-4xl font-extrabold text-emerald-800 mb-8">Your Shopping Cart</h2>

        {cartItems.length === 0 ? (
          <div className="bg-white p-10 rounded-xl shadow-lg border border-gray-100 text-center flex flex-col items-center justify-center min-h-[300px]">
            <p className="text-6xl mb-4">🛒</p>
            <p className="text-xl text-gray-600 font-medium">Your cart is currently empty. Start adding some sustainable items!</p>
            <button
              onClick={handleStartShopping}
              className="mt-8 px-8 py-3 bg-emerald-600 text-white font-semibold rounded-lg shadow-md hover:bg-emerald-700 transition-colors duration-300 transform hover:scale-105"
            >
              Start Shopping
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="md:col-span-2 space-y-5">
              {cartItems.map((item) => (
                <div
                  key={item.id} // Ensure 'id' is present for keys, which it should be from the context
                  className="bg-white p-5 rounded-xl shadow-lg border border-gray-100 flex justify-between items-center transform hover:-translate-y-1 transition-all duration-300"
                >
                  <div className="flex items-center">
                    {/* Placeholder for item image/icon */}
                    <div className="w-16 h-16 bg-emerald-50 rounded-lg flex items-center justify-center text-3xl mr-4">
                        🌱
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-gray-800">{item.name}</h3>
                      <p className="text-lg text-emerald-600 font-semibold mt-1">Price: ${item.price.toFixed(2)}</p>
                    </div>
                  </div>
                  <button
                    onClick={() => removeFromCart(item.id)} // Use removeFromCart from context
                    className="text-red-500 text-2xl hover:text-red-700 p-2 rounded-full hover:bg-red-50 transition-colors duration-200"
                    aria-label={`Remove ${item.name} from cart`}
                  >
                    <FaTrash />
                  </button>
                </div>
              ))}
            </div>

            <div className="md:col-span-1 bg-white p-6 rounded-xl shadow-lg border border-gray-100 self-start">
              <h3 className="text-2xl font-bold text-emerald-800 mb-5">Cart Summary</h3>
              <div className="flex justify-between items-center text-xl text-gray-700 border-b pb-4 border-gray-200">
                <span>Items:</span>
                <span className="font-semibold">{cartItems.length}</span>
              </div>
              <div className="flex justify-between items-center text-3xl font-extrabold text-emerald-700 pt-4">
                <span>Total:</span>
                <span>${total.toFixed(2)}</span>
              </div>
              <button className="mt-8 w-full bg-emerald-600 hover:bg-emerald-700 text-white py-4 rounded-lg font-bold text-xl transition-all duration-300 shadow-md hover:shadow-xl transform hover:scale-105">
                Proceed to Checkout
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}