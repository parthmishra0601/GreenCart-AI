// Dashboard.js
import React, { useState, useEffect } from "react";
import { useCart } from "./CartContext";

export default function Dashboard() {
  const { cartItems } = useCart();
  const DEFAULT_WEEKLY_BUDGET = 100;

  const [budget, setBudget] = useState(DEFAULT_WEEKLY_BUDGET);
  const [extraBudget, setExtraBudget] = useState("");

  const total = cartItems.reduce(
    (sum, item) => sum + item.price * (item.quantity || 1),
    0
  );
  const remaining = budget - total;

  useEffect(() => {
    const storedBudget = localStorage.getItem("weeklyBudget");
    const storedResetDate = localStorage.getItem("lastReset");

    const now = new Date();
    const lastReset = storedResetDate ? new Date(storedResetDate) : null;

    if (!storedBudget || !lastReset || now - lastReset > 7 * 24 * 60 * 60 * 1000) {
      localStorage.setItem("weeklyBudget", DEFAULT_WEEKLY_BUDGET);
      localStorage.setItem("lastReset", now.toISOString());
      setBudget(DEFAULT_WEEKLY_BUDGET);
    } else {
      setBudget(parseFloat(storedBudget));
    }
  }, []);

  useEffect(() => {
    if (remaining < -10) {
      alert("Uh oh! Your spending has significantly exceeded your budget.");
    }
  }, [remaining]);

  const handleIncreaseBudget = () => {
    const amount = parseFloat(extraBudget);
    if (isNaN(amount) || amount <= 0) {
      alert("Please enter a valid positive number.");
      return;
    }
    const newBudget = budget + amount;
    setBudget(newBudget);
    localStorage.setItem("weeklyBudget", newBudget.toString());
    setExtraBudget("");
  };

  const getNextResetDate = () => {
    const storedResetDate = localStorage.getItem("lastReset");
    if (!storedResetDate) return null;
    const resetDate = new Date(storedResetDate);
    resetDate.setDate(resetDate.getDate() + 7);
    return resetDate.toLocaleDateString();
  };

  return (
    <div className="flex h-screen w-full bg-gray-50 font-sans overflow-hidden">
      <main className="flex-grow p-8 overflow-y-auto">
        <h2 className="text-4xl font-extrabold text-emerald-800 mb-8">Dashboard Overview</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {/* Budget Summary */}
          <div className="bg-gradient-to-br from-emerald-500 to-emerald-600 p-8 rounded-xl shadow-xl text-white transform hover:scale-[1.02] transition-transform duration-300">
            <h3 className="text-2xl font-bold mb-4">Budget Summary</h3>
            <div className="space-y-3">
              <p className="flex justify-between items-center text-lg">
                <span>Budget:</span>
                <span className="font-semibold text-xl">${budget.toFixed(2)}</span>
              </p>
              <p className="flex justify-between items-center text-lg">
                <span>Spent:</span>
                <span className="font-semibold text-xl">${total.toFixed(2)}</span>
              </p>
              <p className="flex justify-between items-center text-2xl font-extrabold pt-2 border-t border-emerald-400 mt-2">
                <span>Remaining:</span>
                <span className={`${remaining < 0 ? 'text-red-200' : 'text-green-100'}`}>
                  ${remaining.toFixed(2)}
                </span>
              </p>
              <p className="text-sm text-emerald-100 mt-2">
                Next budget reset on: <span className="font-semibold">{getNextResetDate()}</span>
              </p>
              <div className="mt-6 bg-white bg-opacity-90 p-4 rounded-xl shadow-lg text-emerald-900">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-3xl text-yellow-500">💡</div>
                  <div>
                    <p className="text-lg font-semibold">Need More Budget?</p>
                    <p className="text-sm text-gray-700">
                      Add more budget if you really need to buy something essential.
                    </p>
                  </div>
                </div>
                <div className="flex flex-col sm:flex-row items-center gap-3">
                  <input
                    type="number"
                    min="1"
                    step="0.01"
                    placeholder="Enter extra budget (e.g., 5.00)"
                    value={extraBudget}
                    onChange={(e) => setExtraBudget(e.target.value)}
                    className="w-full sm:w-auto flex-grow border border-gray-300 focus:border-emerald-500 rounded-lg p-2 text-base shadow-sm focus:outline-none transition"
                  />
                  <button
                    onClick={handleIncreaseBudget}
                    className="bg-emerald-600 text-white font-semibold px-4 py-2 rounded-lg shadow hover:bg-emerald-700 transition duration-200 w-full sm:w-auto"
                  >
                    Add Extra Budget
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Cart Summary */}
          <div className="bg-white p-8 rounded-xl shadow-xl border border-gray-100 md:col-span-1 lg:col-span-2">
            <h3 className="text-2xl font-bold text-emerald-800 mb-5">Your Current Cart</h3>
            {cartItems.length === 0 ? (
              <p className="text-lg text-gray-600 py-8 text-center bg-gray-50 rounded-lg">
                <span className="text-5xl block mb-2">🛒</span>
                Your cart is currently empty.
              </p>
            ) : (
              <ul className="text-gray-700 space-y-4">
                {cartItems.map((item, index) => (
                  <li key={index} className="flex justify-between items-center bg-gray-50 p-4 rounded-lg border border-gray-100">
                    <span className="text-lg font-medium text-gray-800">
                      {item.greener_alternative} {item.quantity > 1 && `(x${item.quantity})`}
                    </span>
                    <span className="text-lg font-semibold text-emerald-600">
                      ₹{(item.price * (item.quantity || 1)).toFixed(2)}
                    </span>
                  </li>
                ))}
                <li className="flex justify-between items-center text-xl font-bold text-emerald-700 pt-4 border-t border-gray-200 mt-4">
                  <span>Total Cart Value:</span>
                  <span>₹{total.toFixed(2)}</span>
                </li>
              </ul>
            )}
          </div>

          {/* Environmental Impact */}
          <div className="bg-white p-8 rounded-xl shadow-xl border border-gray-100">
            <h3 className="text-2xl font-bold text-emerald-800 mb-5">Your Environmental Impact</h3>
            <ul className="text-lg text-gray-700 space-y-3">
              <li>Plastic saved: <span className="font-semibold ml-2">1.2kg</span></li>
              <li>Water saved: <span className="font-semibold ml-2">30L</span></li>
              <li>CO₂ reduced: <span className="font-semibold ml-2">4.5kg</span></li>
            </ul>
          </div>

          {/* Sustainability Tips */}
          <div className="bg-white p-8 rounded-xl shadow-xl border border-gray-100 lg:col-span-2">
            <h3 className="text-2xl font-bold text-emerald-800 mb-5">Sustainability Tips</h3>
            <ul className="list-disc list-inside text-lg text-gray-700 space-y-2">
              <li>Use <strong>reusable shopping bags</strong> whenever you go out.</li>
              <li>Choose <strong>biodegradable or minimal packaging</strong> products.</li>
              <li><strong>Buy local and seasonal produce</strong> to reduce your carbon footprint.</li>
              <li>Always consider the <strong>product's lifecycle</strong> before buying.</li>
            </ul>
          </div>
        </div>
      </main>
    </div>
  );
}
