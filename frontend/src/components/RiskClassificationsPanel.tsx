'use client';

import { Award, CheckCircle, TrendingUp, AlertTriangle, Info } from 'lucide-react';

interface RiskClass {
  name: string;
  hkrsRange: string;
  description: string;
  premiumImpact: string;
  color: string;
  bgColor: string;
  borderColor: string;
  icon: 'excellent' | 'good' | 'standard' | 'caution' | 'review';
}

const riskClasses: RiskClass[] = [
  {
    name: 'Excellent',
    hkrsRange: '85-100',
    description: 'Exceptional health metrics across all categories. Demonstrates consistent healthy lifestyle behaviors with excellent cardiorespiratory fitness, regular physical activity, optimal sleep patterns, and healthy body composition.',
    premiumImpact: 'Lowest premium rates available. May qualify for preferred plus rates with up to 15% discount.',
    color: 'text-emerald-700',
    bgColor: 'bg-emerald-50',
    borderColor: 'border-emerald-200',
    icon: 'excellent',
  },
  {
    name: 'Very Good',
    hkrsRange: '70-84',
    description: 'Strong health indicators with above-average fitness levels. Shows consistent physical activity patterns and good sleep habits. Minor areas for improvement but overall demonstrates healthy lifestyle.',
    premiumImpact: 'Preferred rates with 5-10% discount from standard pricing.',
    color: 'text-blue-700',
    bgColor: 'bg-blue-50',
    borderColor: 'border-blue-200',
    icon: 'good',
  },
  {
    name: 'Standard Plus',
    hkrsRange: '55-69',
    description: 'Good health metrics that meet or exceed baseline expectations. Demonstrates moderate physical activity and acceptable health patterns. Some lifestyle factors may benefit from improvement.',
    premiumImpact: 'Standard rates with no loading. The baseline premium for coverage.',
    color: 'text-indigo-700',
    bgColor: 'bg-indigo-50',
    borderColor: 'border-indigo-200',
    icon: 'standard',
  },
  {
    name: 'Standard',
    hkrsRange: '40-54',
    description: 'Acceptable health metrics that meet minimum requirements. May show inconsistent activity levels or some health indicators below optimal ranges. Lifestyle improvements recommended.',
    premiumImpact: 'Standard rates may apply, or minor loading of 10-25% depending on specific factors.',
    color: 'text-amber-700',
    bgColor: 'bg-amber-50',
    borderColor: 'border-amber-200',
    icon: 'caution',
  },
  {
    name: 'Substandard',
    hkrsRange: '0-39',
    description: 'Health metrics indicate areas of concern. May have limited physical activity, poor sleep patterns, or other lifestyle factors that increase risk. Manual review required for underwriting decision.',
    premiumImpact: 'May require additional review. Premium loading of 25-100% may apply, or additional information may be requested.',
    color: 'text-rose-700',
    bgColor: 'bg-rose-50',
    borderColor: 'border-rose-200',
    icon: 'review',
  },
];

const categoryWeights = [
  { name: 'Activity', weight: 25, description: 'Daily steps, active energy, movement trends' },
  { name: 'Fitness (VO2 Max)', weight: 20, description: 'Cardiorespiratory fitness level' },
  { name: 'Vitals (Heart)', weight: 20, description: 'Resting heart rate, HRV, heart health' },
  { name: 'Sleep', weight: 15, description: 'Sleep duration, consistency, quality' },
  { name: 'Body Metrics', weight: 10, description: 'BMI, weight trends, body composition' },
  { name: 'Mobility', weight: 10, description: 'Walking speed, steadiness, balance' },
];

export default function RiskClassificationsPanel() {
  const getIcon = (iconType: string) => {
    switch (iconType) {
      case 'excellent':
        return <Award className="w-6 h-6" />;
      case 'good':
        return <TrendingUp className="w-6 h-6" />;
      case 'standard':
        return <CheckCircle className="w-6 h-6" />;
      case 'caution':
        return <AlertTriangle className="w-6 h-6" />;
      case 'review':
        return <Info className="w-6 h-6" />;
      default:
        return <CheckCircle className="w-6 h-6" />;
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center">
            <Award className="w-6 h-6 text-indigo-600" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-slate-900">Risk Classifications</h1>
            <p className="text-sm text-slate-500">Understanding your HealthKit Risk Score (HKRS) and premium categories</p>
          </div>
        </div>
        
        <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4">
          <h2 className="font-medium text-indigo-900 mb-2">How HKRS Works</h2>
          <p className="text-sm text-indigo-800">
            Your HealthKit Risk Score (HKRS) is calculated from your Apple Health data across 6 key health categories. 
            The score ranges from 0-100 and determines your risk classification, which affects your premium rates.
            Higher scores indicate better health metrics and qualify for lower premiums.
          </p>
        </div>
      </div>

      {/* Category Weights */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">Scoring Categories</h2>
        <p className="text-sm text-slate-600 mb-4">
          Your HKRS is calculated as a weighted sum of scores from these six health categories:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {categoryWeights.map((category) => (
            <div key={category.name} className="border border-slate-200 rounded-lg p-4 hover:border-indigo-300 transition-colors">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium text-slate-900">{category.name}</span>
                <span className="text-lg font-bold text-indigo-600">{category.weight}%</span>
              </div>
              <p className="text-xs text-slate-500">{category.description}</p>
              <div className="mt-2 h-2 bg-slate-100 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-indigo-500 rounded-full"
                  style={{ width: `${category.weight}%` }}
                />
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 p-3 bg-slate-50 rounded-lg">
          <p className="text-xs text-slate-600">
            <strong>Age Adjustment:</strong> Your final HKRS is multiplied by an Age Adjustment Factor (AAF) based on your age bracket. 
            Younger applicants (18-34) have AAF of 1.0, while older applicants have slightly lower factors to account for age-related health expectations.
          </p>
        </div>
      </div>

      {/* Risk Classes */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">Risk Classification Tiers</h2>
        <div className="space-y-4">
          {riskClasses.map((riskClass) => (
            <div 
              key={riskClass.name}
              className={`border rounded-lg p-4 ${riskClass.borderColor} ${riskClass.bgColor}`}
            >
              <div className="flex items-start gap-4">
                <div className={`p-2 rounded-lg ${riskClass.color} bg-white/50`}>
                  {getIcon(riskClass.icon)}
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className={`font-semibold text-lg ${riskClass.color}`}>{riskClass.name}</h3>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${riskClass.color} bg-white/70`}>
                      HKRS: {riskClass.hkrsRange}
                    </span>
                  </div>
                  <p className="text-sm text-slate-700 mb-3">{riskClass.description}</p>
                  <div className="flex items-center gap-2 text-sm">
                    <span className="font-medium text-slate-600">Premium Impact:</span>
                    <span className="text-slate-700">{riskClass.premiumImpact}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* How to Improve */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">How to Improve Your Score</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="border border-emerald-200 bg-emerald-50 rounded-lg p-4">
            <h3 className="font-medium text-emerald-800 mb-2">✓ Do</h3>
            <ul className="text-sm text-emerald-700 space-y-1">
              <li>• Aim for 8,000+ steps daily</li>
              <li>• Get 7-9 hours of consistent sleep</li>
              <li>• Exercise 3-5 times per week</li>
              <li>• Maintain a healthy BMI (18.5-24.9)</li>
              <li>• Track your health data consistently</li>
            </ul>
          </div>
          <div className="border border-amber-200 bg-amber-50 rounded-lg p-4">
            <h3 className="font-medium text-amber-800 mb-2">✗ Avoid</h3>
            <ul className="text-sm text-amber-700 space-y-1">
              <li>• Prolonged sedentary periods</li>
              <li>• Irregular sleep schedules</li>
              <li>• Gaps in health data tracking</li>
              <li>• Ignoring heart rate alerts</li>
              <li>• Sudden changes in activity levels</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
