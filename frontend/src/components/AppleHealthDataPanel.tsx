'use client';

import { 
  Activity, 
  Heart, 
  Moon, 
  Scale, 
  Wind,
  Footprints,
  Dumbbell,
  TrendingUp,
  TrendingDown,
  Minus,
  Smartphone
} from 'lucide-react';

interface AppleHealthDataPanelProps {
  application: any;
}

interface CategoryCardProps {
  title: string;
  weight: string;
  icon: any;
  color: string;
  metrics: { label: string; value: string | number; unit?: string }[];
  trend?: string;
}

function CategoryCard({ title, weight, icon: Icon, color, metrics, trend }: CategoryCardProps) {
  const colorClasses: Record<string, string> = {
    blue: 'bg-blue-50 border-blue-200',
    purple: 'bg-purple-50 border-purple-200',
    red: 'bg-red-50 border-red-200',
    indigo: 'bg-indigo-50 border-indigo-200',
    green: 'bg-green-50 border-green-200',
    amber: 'bg-amber-50 border-amber-200',
    orange: 'bg-orange-50 border-orange-200',
  };

  const iconColors: Record<string, string> = {
    blue: 'text-blue-600 bg-blue-100',
    purple: 'text-purple-600 bg-purple-100',
    red: 'text-red-600 bg-red-100',
    indigo: 'text-indigo-600 bg-indigo-100',
    green: 'text-green-600 bg-green-100',
    amber: 'text-amber-600 bg-amber-100',
    orange: 'text-orange-600 bg-orange-100',
  };

  const getTrendIcon = () => {
    if (!trend) return null;
    const t = trend.toLowerCase().replace(/_/g, ' ');
    if (t.includes('improving') || t.includes('stable')) {
      return <TrendingUp className="w-3 h-3 text-green-500" />;
    }
    if (t.includes('declining') || t.includes('increase')) {
      return <TrendingDown className="w-3 h-3 text-red-500" />;
    }
    return <Minus className="w-3 h-3 text-gray-400" />;
  };

  return (
    <div className={`rounded-xl border-2 p-4 ${colorClasses[color] || colorClasses.blue}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={`p-2 rounded-lg ${iconColors[color] || iconColors.blue}`}>
            <Icon className="w-4 h-4" />
          </div>
          <span className="font-semibold text-gray-900">{title}</span>
        </div>
        <div className="flex items-center gap-2">
          {getTrendIcon()}
          <span className="text-xs font-medium text-gray-500 bg-white px-2 py-0.5 rounded-full">
            {weight}
          </span>
        </div>
      </div>
      <div className="space-y-2">
        {metrics.map((metric, idx) => (
          <div key={idx} className="flex items-center justify-between text-sm">
            <span className="text-gray-600">{metric.label}</span>
            <span className="font-medium text-gray-900">
              {metric.value}
              {metric.unit && <span className="text-gray-400 text-xs ml-1">{metric.unit}</span>}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function AppleHealthDataPanel({ application }: AppleHealthDataPanelProps) {
  // Extract health metrics from llm_outputs
  const healthMetrics = application?.llm_outputs?.health_metrics || {};
  
  const activity = healthMetrics.activity || {};
  const fitness = healthMetrics.fitness || {};
  const heartRate = healthMetrics.heart_rate || {};
  const sleep = healthMetrics.sleep || {};
  const bodyMetrics = healthMetrics.body_metrics || {};
  const mobility = healthMetrics.mobility || {};
  const exercise = healthMetrics.exercise || {};

  const formatNumber = (val: any, decimals = 0) => {
    if (val === undefined || val === null) return '—';
    const num = Number(val);
    if (isNaN(num)) return String(val);
    return decimals > 0 ? num.toFixed(decimals) : num.toLocaleString();
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-red-100 to-pink-100 rounded-xl">
            <Smartphone className="w-6 h-6 text-red-500" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Apple Health Data</h2>
            <p className="text-sm text-gray-500">7 categories from HealthKit</p>
          </div>
        </div>
        <span className="px-3 py-1 bg-green-100 text-green-700 text-xs font-medium rounded-full">
          Data Synced
        </span>
      </div>

      {/* Category Grid - 7 Categories */}
      <div className="space-y-4">
        {/* Row 1: Activity, Fitness, Vitals */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <CategoryCard
            title="Activity"
            weight="25%"
            icon={Activity}
            color="blue"
            trend={activity.trend_6mo}
            metrics={[
              { label: "Daily Steps", value: formatNumber(activity.daily_steps_avg), unit: "" },
              { label: "Active Energy", value: formatNumber(activity.active_energy_burned_avg), unit: "kcal" },
              { label: "Days Tracked", value: formatNumber(activity.days_with_data), unit: "" },
            ]}
          />
          
          <CategoryCard
            title="Fitness"
            weight="20%"
            icon={Wind}
            color="purple"
            metrics={[
              { label: "VO2 Max", value: formatNumber(fitness.vo2_max, 1), unit: "mL/kg/min" },
              { label: "Fitness Level", value: fitness.cardio_fitness_level || "—", unit: "" },
              { label: "Readings", value: formatNumber(fitness.vo2_max_readings), unit: "" },
            ]}
          />
          
          <CategoryCard
            title="Vitals"
            weight="20%"
            icon={Heart}
            color="red"
            metrics={[
              { label: "Resting HR", value: formatNumber(heartRate.resting_hr_avg), unit: "bpm" },
              { label: "HRV", value: formatNumber(heartRate.hrv_avg_ms), unit: "ms" },
              { label: "Irregular Events", value: formatNumber(heartRate.irregular_rhythm_events), unit: "" },
            ]}
          />
        </div>

        {/* Row 2: Sleep, Body Metrics, Mobility, Exercise */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <CategoryCard
            title="Sleep"
            weight="15%"
            icon={Moon}
            color="indigo"
            metrics={[
              { label: "Duration", value: formatNumber(sleep.avg_sleep_duration_hours, 1), unit: "hrs" },
              { label: "Consistency", value: `±${formatNumber(sleep.sleep_consistency_variance_hours, 1)}`, unit: "hrs" },
              { label: "Nights Tracked", value: formatNumber(sleep.nights_with_data), unit: "" },
            ]}
          />
          
          <CategoryCard
            title="Body Metrics"
            weight="10%"
            icon={Scale}
            color="green"
            trend={bodyMetrics.bmi_trend}
            metrics={[
              { label: "BMI", value: formatNumber(bodyMetrics.bmi, 1), unit: "" },
              { label: "Weight", value: formatNumber(bodyMetrics.weight_kg, 1), unit: "kg" },
              { label: "Trend", value: (bodyMetrics.bmi_trend || "stable").replace(/_/g, " "), unit: "" },
            ]}
          />
          
          <CategoryCard
            title="Mobility"
            weight="10%"
            icon={Footprints}
            color="amber"
            metrics={[
              { label: "Walk Speed", value: formatNumber(mobility.walking_speed_avg, 2), unit: "m/s" },
              { label: "Steadiness", value: (mobility.walking_steadiness || "normal").replace(/_/g, " "), unit: "" },
            ]}
          />
          
          <CategoryCard
            title="Exercise"
            weight="—"
            icon={Dumbbell}
            color="orange"
            metrics={[
              { label: "Weekly Sessions", value: formatNumber(exercise.workout_frequency_weekly), unit: "" },
              { label: "Avg Duration", value: formatNumber(exercise.workout_avg_duration_minutes), unit: "min" },
            ]}
          />
        </div>
      </div>

      {/* Data Quality Footer */}
      <div className="mt-6 pt-4 border-t border-gray-100">
        <div className="flex items-center justify-between text-sm text-gray-500">
          <span>Data source: Apple HealthKit</span>
          <span>
            {activity.days_with_data >= 90 ? '✓ High data quality' : 
             activity.days_with_data >= 30 ? '○ Medium data quality' : '⚠ Limited data'}
          </span>
        </div>
      </div>
    </div>
  );
}
