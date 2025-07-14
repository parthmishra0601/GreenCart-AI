import React, { useState, useEffect } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import {
  FaSpinner,
  FaShoppingCart,
  FaCheckCircle,
  FaTimesCircle,
  FaPlus,
  FaMinus,
  FaTrash,
} from "react-icons/fa";
import { useCart } from "./CartContext";

export default function Home() {
  const navigate = useNavigate();
  const {
    cartItems = [],
    addToCart,
    removeFromCart,
    updateQuantity,
    clearCart,
  } = useCart();

  const [productName, setProductName] = useState("");
  const [categories, setCategories] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState("");
  const [suggestion, setSuggestion] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showCart, setShowCart] = useState(false);
  const [toastMessage, setToastMessage] = useState({ text: "", type: "" });
  const [showClearCartConfirm, setShowClearCartConfirm] = useState(false);

  const showToast = (text, type = "success") => {
    setToastMessage({ text, type });
    setTimeout(() => {
      setToastMessage({ text: "", type: "" });
    }, 3000);
  };

  useEffect(() => {
    axios
      .get("http://localhost:5000/categories")
      .then((res) => setCategories(res.data))
      .catch((err) => {
        console.error("Error fetching categories:", err);
        setError("Failed to load categories.");
      });
  }, []);

  const handleSuggest = async () => {
    setError("");
    setSuggestion(null);
    setLoading(true);

    if (!productName || !selectedCategory) {
      setError("Please provide both product name and category.");
      setLoading(false);
      return;
    }

    try {
      const response = await axios.post("http://localhost:5000/recommend", {
        product: productName,
        category: selectedCategory,
      });

      if (response.data.greener_alternative) {
        setSuggestion(response.data);
        showToast("Greener alternative found!", "success");
      } else {
        setSuggestion({ message: response.data.message });
        showToast(response.data.message, "info");
      }
    } catch (err) {
      console.error(err);
      setError("Error fetching greener alternative.");
      showToast("Error during recommendation.", "error");
    } finally {
      setLoading(false);
    }
  };

  const getTotalItemsInCart = () =>
    cartItems.reduce((total, item) => total + item.quantity, 0);

  const getTotalCartValue = () =>
    cartItems
      .reduce(
        (total, item) =>
          total + (parseFloat(item.price) || 0) * (item.quantity || 1),
        0
      )
      .toFixed(2);

  const handleCheckout = () => {
    showToast("Redirecting to Dashboard...", "info");
    setTimeout(() => {
      navigate("/dashboard");
    }, 1000);
  };

  return (
    <div className="min-h-screen bg-green-50 p-4 flex items-center justify-center relative">
      {/* Toast Notification */}
      {toastMessage.text && (
        <div
          className={`fixed top-5 left-1/2 -translate-x-1/2 px-6 py-3 rounded-lg shadow-xl text-white font-semibold flex items-center gap-2 z-50
          ${
            toastMessage.type === "success"
              ? "bg-green-500"
              : toastMessage.type === "error"
              ? "bg-red-500"
              : "bg-blue-500"
          }`}
        >
          {toastMessage.type === "success" && <FaCheckCircle />}
          {toastMessage.type === "error" && <FaTimesCircle />}
          {toastMessage.text}
        </div>
      )}

      <div className="max-w-2xl w-full bg-white rounded-xl shadow-lg p-6 relative">
        <h1 className="text-3xl font-extrabold mb-4 text-center text-green-800">
          🌱 Greener Product Recommender
        </h1>

        {/* Cart Icon */}
        <div className="absolute top-4 right-4">
          <button
            onClick={() => setShowCart(!showCart)}
            className="relative p-2 bg-emerald-600 text-white rounded-full"
          >
            <FaShoppingCart />
            {getTotalItemsInCart() > 0 && (
              <span className="absolute -top-1 -right-1 bg-red-600 text-xs rounded-full px-2">
                {getTotalItemsInCart()}
              </span>
            )}
          </button>
        </div>

        {/* Input Fields */}
        <div className="space-y-4 mb-6">
          <input
            type="text"
            placeholder="Enter product name"
            value={productName}
            onChange={(e) => setProductName(e.target.value)}
            className="w-full border px-4 py-2 rounded"
          />

          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="w-full border px-4 py-2 rounded"
          >
            <option value="">Select Category</option>
            {categories.map((cat, index) => (
              <option key={index} value={cat}>
                {cat}
              </option>
            ))}
          </select>

          <button
            onClick={handleSuggest}
            disabled={loading}
            className="w-full bg-green-600 text-white py-2 rounded"
          >
            {loading ? "Loading..." : "Suggest Greener Alternative"}
          </button>
        </div>

        {error && <p className="text-red-500">{error}</p>}

        {/* Suggestion Result */}
        {suggestion && !suggestion.message && (
          <div className="mt-4 p-4 bg-green-100 rounded">
            <h2 className="font-bold text-green-700 text-xl mb-2">Suggestion:</h2>
            <p>
              <strong>Alternative:</strong> {suggestion.greener_alternative}
            </p>
            <p>
              <strong>Eco Score:</strong> {suggestion.eco_score}
            </p>
            <p>
              <strong>Price:</strong> ₹{suggestion.price}
            </p>
            <button
              onClick={() => addToCart(suggestion)}
              className="mt-2 bg-lime-600 text-white px-4 py-2 rounded"
            >
              Add to Cart
            </button>
          </div>
        )}

        {suggestion?.message && (
          <div className="mt-4 p-4 bg-blue-100 rounded text-blue-700">
            {suggestion.message}
          </div>
        )}
      </div>

      {/* Shopping Cart Side Panel */}
      {showCart && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-end z-50">
          <div className="bg-white w-full max-w-sm h-full p-6 overflow-y-auto relative">
            <button
              onClick={() => setShowCart(false)}
              className="absolute top-4 right-4 text-2xl font-bold"
            >
              &times;
            </button>
            <h2 className="text-xl font-bold mb-4">Your Cart</h2>
            {cartItems.length === 0 ? (
              <p>Your cart is empty.</p>
            ) : (
              <>
                {cartItems.map((item, index) => (
                  <div
                    key={index}
                    className="mb-4 border-b pb-2 flex justify-between items-center"
                  >
                    <div>
                      <p className="font-semibold">{item.greener_alternative}</p>
                      <p className="text-sm text-gray-500">
                        Quantity: {item.quantity} | ₹{item.price}
                      </p>
                    </div>
                    <div className="flex gap-1">
                      <button
                        onClick={() =>
                          updateQuantity(item.greener_alternative, -1)
                        }
                        className="px-2 py-1 bg-gray-200 rounded"
                      >
                        <FaMinus />
                      </button>
                      <button
                        onClick={() =>
                          updateQuantity(item.greener_alternative, 1)
                        }
                        className="px-2 py-1 bg-gray-200 rounded"
                      >
                        <FaPlus />
                      </button>
                      <button
                        onClick={() =>
                          removeFromCart(item.greener_alternative)
                        }
                        className="px-2 py-1 bg-red-200 rounded"
                      >
                        <FaTrash />
                      </button>
                    </div>
                  </div>
                ))}

                <div className="text-right font-bold mb-4">
                  Total: ₹{getTotalCartValue()}
                </div>

                <button
                  onClick={handleCheckout}
                  className="w-full bg-green-600 text-white py-2 rounded mb-2"
                >
                  Proceed to Checkout
                </button>
                <button
                  onClick={() => setShowClearCartConfirm(true)}
                  className="w-full bg-red-500 text-white py-2 rounded"
                >
                  Clear Cart
                </button>
              </>
            )}
          </div>
        </div>
      )}

      {/* Confirm Clear Cart */}
      {showClearCartConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded shadow-md">
            <p>Are you sure you want to clear the cart?</p>
            <div className="flex justify-end mt-4 gap-3">
              <button
                onClick={() => {
                  clearCart();
                  setShowClearCartConfirm(false);
                  showToast("Cart cleared.", "info");
                }}
                className="bg-red-500 text-white px-4 py-2 rounded"
              >
                Yes
              </button>
              <button
                onClick={() => setShowClearCartConfirm(false)}
                className="bg-gray-300 px-4 py-2 rounded"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
