// src/Components/SustainabilityTips.jsx
import React from "react";
import { FiAlertCircle } from "react-icons/fi"; // Icon import from react-icons

const tips = [
  {
    title: "Use Reusable Shopping Bags",
    description: "Avoid plastic bags by carrying your own cloth or jute bags.",
  },
  {
    title: "Choose Biodegradable Packaging",
    description: "Opt for products with eco-friendly packaging that breaks down naturally.",
  },
  {
    title: "Buy Local and Seasonal Products",
    description: "Reduce carbon footprint by supporting local farmers and seasonal goods.",
  },
  {
    title: "Reduce Food Waste",
    description: "Plan meals ahead and use leftovers creatively.",
  },
  {
    title: "Conserve Water",
    description: "Fix leaky taps and use water-saving appliances.",
  },
  {
    title: "Recycle and Compost",
    description: "Separate recyclables and compost organic waste to reduce landfill pollution.",
  },
  {
    title: "Switch to LED Lighting",
    description: "Energy-efficient LED bulbs last longer and reduce electricity consumption.",
  },
  {
    title: "Unplug Electronics",
    description: "Avoid 'vampire' energy drain by unplugging chargers and devices when not in use.",
  },
];

export default function SustainabilityTips() {
  return (
    <div className="flex min-h-screen w-full bg-gray-50 font-sans overflow-hidden">
      <main className="flex-grow p-8 overflow-y-auto">
        <h2 className="text-4xl font-extrabold text-emerald-800 mb-8">
          Sustainability Tips
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {tips.map((tip, index) => (
            <div
              key={index}
              className="bg-white p-6 rounded-xl shadow-lg border border-gray-100 hover:shadow-xl transform hover:-translate-y-1 transition-all duration-300 group"
            >
              <div className="flex items-center mb-3">
                <FiAlertCircle className="text-2xl mr-3 text-emerald-500 group-hover:scale-110 transition-transform duration-300" />
                <h3 className="text-xl font-bold text-gray-800">
                  {tip.title}
                </h3>
              </div>
              <p className="text-lg text-gray-700 leading-relaxed">{tip.description}</p>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}
