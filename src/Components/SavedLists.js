// Components/SavedLists.jsx
import React from 'react';
import { useCart } from './CartContext'; // Adjust path if necessary
import { FaHeart } from 'react-icons/fa'; // Import the heart icon

export default function SavedLists() {
  // Access favoriteItems and toggleFavorite from your global context
  const { favoriteItems, toggleFavorite } = useCart();

  return (
    <div className="flex-grow p-8 bg-gray-50 min-h-screen">
      <h1 className="text-4xl font-extrabold text-emerald-800 mb-8">Your Saved Lists</h1>

      {/* Favorites Section */}
      <div className="bg-white p-6 shadow-xl rounded-xl border border-gray-100 mb-8">
        <h2 className="text-3xl font-bold text-emerald-800 mb-6 flex items-center">
          <FaHeart className="text-red-500 mr-3 text-4xl" /> Your Favorite Items
        </h2>

        {favoriteItems.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {favoriteItems.map((item) => (
              <div
                key={item.id}
                className="bg-white p-5 shadow-lg rounded-xl border border-gray-100 flex flex-col items-start relative"
              >
                <div className="relative h-24 bg-emerald-50 rounded-lg mb-4 w-full flex items-center justify-center overflow-hidden">
                  <span className="text-emerald-300 text-5xl">💖</span>
                  <button
                    onClick={() => toggleFavorite(item)} // Allow removing from favorites here
                    className="absolute top-2 right-2 p-2 rounded-full text-red-500 hover:text-red-600 transition-colors duration-200"
                    title="Remove from Favorites"
                  >
                    <FaHeart className="text-xl" />
                  </button>
                </div>
                <h3 className="text-lg font-bold text-gray-800 mb-1">{item.name}</h3>
                <p className="text-md text-emerald-600 font-semibold">Price: ${item.price.toFixed(2)}</p>
                <p className="text-sm text-gray-500 mt-1">{item.category}</p>
                {item.isSustainable && (
                  <span className="mt-2 inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800">
                    Sustainable 🌱
                  </span>
                )}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xl text-gray-600 p-4 text-center border-2 border-dashed border-gray-200 rounded-lg">
            You haven't saved any items to your favorites yet. Go to the Home page and click the <FaHeart className="inline text-red-400"/> icon!
          </p>
        )}
      </div>

      {/* Placeholder for other saved lists */}
      <div className="bg-white p-6 shadow-xl rounded-xl border border-gray-100 mt-8">
        <h2 className="text-3xl font-bold text-emerald-800 mb-6">Your Custom Shopping Lists</h2>
        <p className="text-xl text-gray-600 p-4 text-center border-2 border-dashed border-gray-200 rounded-lg">
          This section can be expanded to allow users to create and manage their own shopping lists.
        </p>
      </div>
    </div>
  );
}