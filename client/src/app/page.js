'use client';

import React, { useState, useCallback, useEffect } from 'react';
import { MapPin, Clock, Cloud, Car, Calendar, Pizza, AlertCircle, CheckCircle, Send } from 'lucide-react';
import axios from 'axios';

const PizzaDeliveryPredictor = () => {
  const [formData, setFormData] = useState({
    distance_miles: '',
    pizza_count: 1,
    day_of_week: '',
    weather: '',
    traffic_level: ''
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [cache, setCache] = useState({});

  const API_BASE_URL =  'http://localhost:8000';
  const CACHE_DURATION = 30 * 60 * 1000; // 30 minutes

  // Form options
  const dayOptions = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
  const weatherOptions = ['Sunny', 'Cloudy', 'Rainy', 'Snowy'];
  const trafficOptions = ['Light', 'Medium', 'Heavy'];

  // Cache management
  const getCacheKey = useCallback((data) => {
    const hour = new Date().getHours();
    return `${data.distance_miles}-${data.pizza_count}-${data.day_of_week}-${data.weather}-${data.traffic_level}-${hour}`;
  }, []);

  const isCacheValid = useCallback((timestamp) => {
    return Date.now() - timestamp < CACHE_DURATION;
  }, [CACHE_DURATION]);

  const getCachedPrediction = useCallback((data) => {
    const key = getCacheKey(data);
    const cached = cache[key];
    return cached && isCacheValid(cached.timestamp) ? cached.data : null;
  }, [cache, getCacheKey, isCacheValid]);

  const setCachedPrediction = useCallback((data, result) => {
    const key = getCacheKey(data);
    setCache(prev => ({
      ...prev,
      [key]: { data: result, timestamp: Date.now() }
    }));
  }, [getCacheKey]);

  // Clean expired cache entries
  useEffect(() => {
    const cleanupInterval = setInterval(() => {
      setCache(prev => {
        const cleaned = {};
        Object.entries(prev).forEach(([key, value]) => {
          if (isCacheValid(value.timestamp)) {
            cleaned[key] = value;
          }
        });
        return cleaned;
      });
    }, 5 * 60 * 1000); // Clean every 5 minutes

    return () => clearInterval(cleanupInterval);
  }, [isCacheValid]);

  // Auto-fill current day
  useEffect(() => {
    if (!formData.day_of_week) {
      const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
      const currentDay = days[new Date().getDay()];
      setFormData(prev => ({ ...prev, day_of_week: currentDay }));
    }
  }, [formData.day_of_week]);

  // Handle form input changes
  const handleInputChange = useCallback((field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (error) setError(''); // Clear error when user makes changes
  }, [error]);

  // Validate form data
  const validateForm = useCallback(() => {
    const errors = [];

    if (!formData.distance_miles || formData.distance_miles <= 0) {
      errors.push('Distance must be greater than 0 miles');
    }

    if (formData.distance_miles > 50) {
      errors.push('Distance cannot exceed 50 miles');
    }

    if (formData.pizza_count < 1 || formData.pizza_count > 20) {
      errors.push('Pizza count must be between 1 and 20');
    }

    if (!formData.day_of_week) {
      errors.push('Please select a day of week');
    }

    if (!formData.weather) {
      errors.push('Please select weather condition');
    }

    if (!formData.traffic_level) {
      errors.push('Please select traffic level');
    }

    return errors;
  }, [formData]);

  // Submit prediction request
  const handleSubmit = async (e) => {
    e.preventDefault();

    const validationErrors = validateForm();
    if (validationErrors.length > 0) {
      setError(validationErrors[0]);
      return;
    }

    // Check cache first
    const cachedResult = getCachedPrediction(formData);
    if (cachedResult) {
      setPrediction(cachedResult);
      setError('');
      return;
    }

    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      const requestData = {
        distance_miles: parseFloat(formData.distance_miles),
        pizza_count: parseInt(formData.pizza_count),
        day_of_week: formData.day_of_week,
        weather: formData.weather,
        traffic_level: formData.traffic_level,
      };

      console.log('Sending prediction request:', requestData);

      const response = await axios.post(`${API_BASE_URL}/predict`, requestData, {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 10000, // 10 second timeout
      });

      console.log('Prediction response:', response.data);

      // Transform backend response to frontend format
      const transformedPrediction = {
        predicted_delivery_time: response.data.predicted_time_minutes,
        distance_miles: requestData.distance_miles,
        weather_condition: requestData.weather,
        traffic_level: requestData.traffic_level,
        day_of_week: requestData.day_of_week,
        pizza_count: requestData.pizza_count,
        confidence_score: 0.85, // You can enhance this based on your model
        message: response.data.formatted_message,
        order_details: response.data.order_details
      };

      setPrediction(transformedPrediction);
      setCachedPrediction(formData, transformedPrediction);

    } catch (err) {
      console.error('Prediction error:', err);

      let errorMessage = 'Failed to get delivery prediction. Please try again.';

      if (err.code === 'ECONNABORTED') {
        errorMessage = 'Request timeout. Please check your connection and try again.';
      } else if (err.code === 'ERR_NETWORK') {
        errorMessage = 'Cannot connect to server. Please ensure the backend is running on ' + API_BASE_URL;
      } else if (err.response) {
        errorMessage = err.response.data?.detail || `Server error: ${err.response.status}`;
      } else if (err.message) {
        errorMessage = err.message;
      }

      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // Reset form
  const resetForm = () => {
    setFormData({
      distance_miles: '',
      pizza_count: 1,
      day_of_week: dayOptions[new Date().getDay()],
      weather: '',
      traffic_level: ''
    });
    setPrediction(null);
    setError('');
  };

  // Helper functions for styling
  const getConfidenceColor = (score) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getTrafficColor = (level) => {
    switch (level?.toLowerCase()) {
      case 'light': return 'text-green-600';
      case 'medium': return 'text-yellow-600';
      case 'heavy': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 to-red-50 p-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex justify-center mb-4">
            <Pizza className="w-16 h-16 text-orange-500" />
          </div>
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            🍕 Pizza Delivery Time Predictor
          </h1>
          <p className="text-gray-600">
            Enter delivery details to get AI-powered delivery time predictions
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Input Form */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-semibold flex items-center text-black">
                <Pizza className="w-6 h-6 mr-2 text-orange-500" />
                Order Details
              </h2>
              <button
                onClick={resetForm}
                className="text-sm text-gray-500 hover:text-gray-700 underline"
              >
                Reset Form
              </button>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Distance */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <MapPin className="w-4 h-4 inline mr-1" />
                  Distance (miles)
                </label>
                <input
                  type="number"
                  step="0.1"
                  min="0.1"
                  max="50"
                  value={formData.distance_miles}
                  onChange={(e) => handleInputChange('distance_miles', e.target.value)}
                  placeholder="e.g., 2.5"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent text-gray-500"
                  disabled={loading}
                />
                <p className="text-xs text-gray-500 mt-1">Distance from restaurant to delivery location</p>
              </div>

              {/* Pizza Count */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Pizza className="w-4 h-4 inline mr-1" />
                  Number of Pizzas
                </label>
                <div className="flex items-center space-x-3">
                  <button
                    type="button"
                    onClick={() => handleInputChange('pizza_count', Math.max(1, formData.pizza_count - 1))}
                    className="w-10 h-10 rounded-full bg-orange-100 text-orange-600 hover:bg-orange-200 transition-colors disabled:opacity-50"
                    disabled={formData.pizza_count <= 1 || loading}
                  >
                    -
                  </button>
                  <span className="text-2xl font-bold w-12 text-center text-black">{formData.pizza_count}</span>
                  <button
                    type="button"
                    onClick={() => handleInputChange('pizza_count', Math.min(20, formData.pizza_count + 1))}
                    className="w-10 h-10 rounded-full bg-orange-100 text-orange-600 hover:bg-orange-200 transition-colors disabled:opacity-50"
                    disabled={formData.pizza_count >= 20 || loading}
                  >
                    +
                  </button>
                </div>
              </div>

              {/* Day of Week */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Calendar className="w-4 h-4 inline mr-1" />
                  Day of Week
                </label>
                <select
                  value={formData.day_of_week}
                  onChange={(e) => handleInputChange('day_of_week', e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent text-black"
                  disabled={loading}
                >
                  <option value="">Select day</option>
                  {dayOptions.map(day => (
                    <option key={day} value={day}>{day}</option>
                  ))}
                </select>
              </div>

              {/* Weather */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Cloud className="w-4 h-4 inline mr-1" />
                  Weather Condition
                </label>
                <select
                  value={formData.weather}
                  onChange={(e) => handleInputChange('weather', e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent text-black"
                  disabled={loading}
                >
                  <option value="">Select weather</option>
                  {weatherOptions.map(weather => (
                    <option key={weather} value={weather}>{weather}</option>
                  ))}
                </select>
              </div>

              {/* Traffic */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Car className="w-4 h-4 inline mr-1" />
                  Traffic Level
                </label>
                <select
                  value={formData.traffic_level}
                  onChange={(e) => handleInputChange('traffic_level', e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent text-black"
                  disabled={loading}
                >
                  <option value="">Select traffic</option>
                  {trafficOptions.map(traffic => (
                    <option key={traffic} value={traffic}>{traffic}</option>
                  ))}
                </select>
              </div>

              {/* Error Message */}
              {error && (
                <div className="flex items-center space-x-2 text-red-600 bg-red-50 p-3 rounded-lg">
                  <AlertCircle className="w-5 h-5 flex-shrink-0" />
                  <span className="text-sm">{error}</span>
                </div>
              )}

              {/* Submit Button */}
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-orange-500 text-white py-3 px-6 rounded-lg font-semibold hover:bg-orange-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center space-x-2"
              >
                {loading ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    <span>Predicting...</span>
                  </>
                ) : (
                  <>
                    <Send className="w-5 h-5" />
                    <span>Get Prediction</span>
                  </>
                )}
              </button>
            </form>
          </div>

          {/* Prediction Results */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-semibold mb-6 flex items-center">
              <Clock className="w-6 h-6 mr-2 text-blue-500" />
              Delivery Prediction
            </h2>

            {prediction ? (
              <div className="space-y-6">
                {/* Main Prediction */}
                <div className="text-center bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-6">
                  <div className="text-4xl font-bold text-blue-600 mb-2">
                    {Math.round(prediction.predicted_delivery_time)} min
                  </div>
                  <p className="text-gray-700 font-medium">
                    {prediction.message}
                  </p>
                </div>

                {/* Details Grid */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <MapPin className="w-4 h-4 text-gray-600" />
                      <span className="text-sm font-medium text-gray-600">Distance</span>
                    </div>
                    <div className="text-lg font-bold text-gray-800">
                      {prediction.distance_miles} miles
                    </div>
                  </div>

                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <Pizza className="w-4 h-4 text-gray-600" />
                      <span className="text-sm font-medium text-gray-600">Pizzas</span>
                    </div>
                    <div className="text-lg font-bold text-gray-800">
                      {prediction.pizza_count}
                    </div>
                  </div>

                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <Cloud className="w-4 h-4 text-gray-600" />
                      <span className="text-sm font-medium text-gray-600">Weather</span>
                    </div>
                    <div className="text-lg font-bold text-gray-800">
                      {prediction.weather_condition}
                    </div>
                  </div>

                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <Car className="w-4 h-4 text-gray-600" />
                      <span className="text-sm font-medium text-gray-600">Traffic</span>
                    </div>
                    <div className={`text-lg font-bold ${getTrafficColor(prediction.traffic_level)}`}>
                      {prediction.traffic_level}
                    </div>
                  </div>
                </div>

                {/* Day and Confidence */}
                <div className="space-y-4">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <Calendar className="w-4 h-4 text-gray-600" />
                      <span className="text-sm font-medium text-gray-600">Day of Week</span>
                    </div>
                    <div className="text-lg font-bold text-gray-800">
                      {prediction.day_of_week}
                    </div>
                  </div>

                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <CheckCircle className="w-4 h-4 text-gray-600" />
                        <span className="text-sm font-medium text-gray-600">Confidence</span>
                      </div>
                      <span className={`text-sm font-bold ${getConfidenceColor(prediction.confidence_score)}`}>
                        {Math.round(prediction.confidence_score * 100)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full transition-all duration-300 ${prediction.confidence_score >= 0.8 ? 'bg-green-500' :
                            prediction.confidence_score >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                          }`}
                        style={{ width: `${prediction.confidence_score * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-500 py-12">
                <Clock className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                <p className="text-lg mb-2">Ready for Prediction</p>
                <p className="text-sm">
                  Fill out the form and click &quot;Get Prediction&quot; to see results
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Cache Status & Server Info */}
        <div className="mt-6 flex justify-between items-center text-sm text-gray-500">
          <div>
            {Object.keys(cache).length > 0 && (
              <span>📋 {Object.keys(cache).length} prediction(s) cached</span>
            )}
          </div>
          <div>
            🔗 Server: {API_BASE_URL}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PizzaDeliveryPredictor;