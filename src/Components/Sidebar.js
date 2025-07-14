import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { FaHome, FaChartBar, FaLeaf, FaSignOutAlt } from "react-icons/fa";

const navItems = [
  { label: "Home", icon: <FaHome />, path: "/" },
  { label: "Dashboard", icon: <FaChartBar />, path: "/dashboard" },
  { label: "Sustainability Tips", icon: <FaLeaf />, path: "/tips" },
];

export default function Sidebar() {
  const navigate = useNavigate();

  const handleLogout = () => {
    console.log("User logged out!");
    navigate("/login");
  };

  return (
    <aside className="min-h-screen w-64 bg-emerald-700 shadow-2xl flex flex-col text-white sticky top-0">
      {/* Top section */}
      <div className="px-6 pt-6 flex-1 flex flex-col">
        <h1 className="text-3xl font-extrabold text-white mb-8 mt-2">
          GreenCart AI
        </h1>
        <nav className="space-y-3 flex-1">
          {navItems.map((item) => (
            <Link key={item.path} to={item.path} className="block">
              <div className="flex items-center p-4 rounded-xl transition-all duration-300 cursor-pointer text-emerald-100 hover:bg-emerald-600 hover:text-white transform hover:translate-x-1">
                <span className="text-2xl mr-4">{item.icon}</span>
                <span className="text-lg font-medium">{item.label}</span>
              </div>
            </Link>
          ))}
        </nav>
      </div>

      {/* Bottom Logout section */}
      <div className="px-6 pb-6">
        <button
          onClick={handleLogout}
          className="w-full bg-emerald-600 hover:bg-emerald-800 text-white py-3 rounded-lg transition-all duration-300 font-semibold text-lg shadow-lg hover:shadow-xl flex items-center justify-center space-x-2"
        >
          <FaSignOutAlt className="text-xl" />
          <span>Logout</span>
        </button>
      </div>
    </aside>
  );
}
