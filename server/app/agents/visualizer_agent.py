from typing import List, Tuple, Dict, Any
from langchain_core.messages import AIMessage
import base64

try:
	from vl_convert import vegalite_to_png  # type: ignore
except Exception:  # pragma: no cover
	vegalite_to_png = None  # type: ignore


def _spec_to_png_data_uri(spec: Dict[str, Any]):
	try:
		if vegalite_to_png is None:
			return None
		png_bytes = vegalite_to_png(spec)  # type: ignore
		b64 = base64.b64encode(png_bytes).decode("ascii")
		return f"data:image/png;base64,{b64}"
	except Exception:
		return None


def create_visualizer_node():
	def visualizer_node(state: Dict[str, Any]) -> Dict[str, Any]:
		steps = state.get("intermediate_steps", []) or []
		risk_data = None
		fraud_data = None
		projection_data = None

		for step in steps:
			try:
				if isinstance(step, (list, tuple)) and len(step) == 2:
					action, observation = step
					tool_name = getattr(action, "tool", None) or getattr(action, "tool_name", None) or ""
					if isinstance(observation, Dict):
						if tool_name == "analyze_risk_profile" and risk_data is None:
							risk_data = observation
						elif tool_name == "detect_fraud" and fraud_data is None:
							fraud_data = observation
						elif tool_name == "project_pension" and projection_data is None:
							projection_data = observation
			except Exception:
				continue

		charts: Dict[str, Any] = {}

		# Projection line (start vs end)
		try:
			if isinstance(projection_data, dict):
				# Check for new comprehensive pension data structure
				if "current_savings" in projection_data and "retirement_goal" in projection_data:
					# New comprehensive structure
					current_savings = projection_data.get("current_savings", "$0")
					retirement_goal = projection_data.get("retirement_goal", "$0")
					progress = projection_data.get("progress_to_goal", "0%")
					status = projection_data.get("status", "Unknown")
					years_remaining = projection_data.get("years_remaining", 0)
					savings_rate = projection_data.get("savings_rate", "0%")
					nominal_projection = projection_data.get("nominal_projection", "$0")
					inflation_adjusted = projection_data.get("inflation_adjusted", False)
					
					# Create progress chart
					charts["pension_progress"] = {
						"$schema": "https://vega.github.io/schema/vega-lite/v5.json",
						"description": "Retirement Goal Progress",
						"data": {"values": [
							{"category": "Current Savings", "amount": float(str(current_savings).replace("$", "").replace(",", ""))},
							{"category": "Remaining Goal", "amount": float(str(retirement_goal).replace("$", "").replace(",", "")) - float(str(current_savings).replace("$", "").replace(",", ""))}
						]},
						"mark": {"type": "bar", "stack": "zero"},
						"encoding": {
							"x": {"field": "category", "type": "nominal", "title": ""},
							"y": {"field": "amount", "type": "quantitative", "title": "Amount ($)"},
							"color": {"field": "category", "type": "nominal"}
						}
					}
					
					# Create progress gauge
					charts["progress_gauge"] = {
						"$schema": "https://vega.github.io/schema/vega-lite/v5.json",
						"description": "Progress to Goal",
						"data": {"values": [{"progress": float(str(progress).replace("%", ""))}]},
						"mark": {"type": "arc", "innerRadius": 50},
						"encoding": {
							"theta": {"field": "progress", "type": "quantitative", "scale": {"domain": [0, 100]}},
							"color": {"field": "progress", "type": "quantitative", "scale": {"range": ["#ff4444", "#ffaa00", "#44ff44"]}}
						}
					}
					
					# Create timeline chart with both nominal and inflation-adjusted projections
					if years_remaining > 0:
						charts["retirement_timeline"] = {
							"$schema": "https://vega.github.io/schema/vega-lite/v5.json",
							"description": "Years to Retirement",
							"data": {"values": [
								{"year": 0, "balance": float(str(current_savings).replace("$", "").replace(",", "")), "type": "Current"},
								{"year": years_remaining, "balance": float(str(projection_data.get("projected_balance_at_retirement", "$0")).replace("$", "").replace(",", "")), "type": "Inflation-Adjusted"},
								{"year": years_remaining, "balance": float(str(nominal_projection).replace("$", "").replace(",", "")), "type": "Nominal"}
							]},
							"mark": {"type": "line", "point": True},
							"encoding": {
								"x": {"field": "year", "type": "quantitative", "title": "Years"},
								"y": {"field": "balance", "type": "quantitative", "title": "Balance ($)"},
								"color": {"field": "type", "type": "nominal"}
							}
						}
					
					# Create comparison chart showing nominal vs inflation-adjusted
					if inflation_adjusted and nominal_projection != "$0":
						charts["projection_comparison"] = {
							"$schema": "https://vega.github.io/schema/vega-lite/v5.json",
							"description": "Nominal vs Inflation-Adjusted Projection",
							"data": {"values": [
								{"type": "Nominal Projection", "amount": float(str(nominal_projection).replace("$", "").replace(",", ""))},
								{"type": "Inflation-Adjusted", "amount": float(str(projection_data.get("projected_balance_at_retirement", "$0")).replace("$", "").replace(",", ""))}
							]},
							"mark": {"type": "bar"},
							"encoding": {
								"x": {"field": "type", "type": "nominal", "title": ""},
								"y": {"field": "amount", "type": "quantitative", "title": "Amount ($)"},
								"color": {"field": "type", "type": "nominal"}
							}
						}
				else:
					# Fallback to old structure
					years = int(projection_data.get("projection_period_years") or 0)
					def to_num(x):
						try:
							return float(str(x).replace("$", "").replace(",", ""))
						except Exception:
							return None
					start_val = to_num(projection_data.get("starting_balance"))
					end_val = to_num(projection_data.get("projected_balance"))
					if start_val is not None and end_val is not None:
						charts["projection"] = {
							"$schema": "https://vega.github.io/schema/vega-lite/v5.json",
							"description": "Projected balance over time",
							"data": {"values": [
								{"year": 0, "balance": start_val},
								{"year": years or 10, "balance": end_val}
							]},
							"mark": {"type": "line", "point": True},
							"encoding": {
								"x": {"field": "year", "type": "quantitative", "title": "Year"},
								"y": {"field": "balance", "type": "quantitative", "title": "Balance ($)"}
							}
						}
		except Exception:
			pass

		# Risk score bar
		try:
			if isinstance(risk_data, dict):
				score = risk_data.get("risk_score")
				if isinstance(score, (int, float)):
					charts["risk"] = {
						"$schema": "https://vega.github.io/schema/vega-lite/v5.json",
						"description": "Risk score",
						"data": {"values": [{"metric": "Risk Score", "value": float(score)}]},
						"mark": "bar",
						"encoding": {
							"x": {"field": "metric", "type": "nominal", "title": ""},
							"y": {"field": "value", "type": "quantitative", "title": "Score"}
						}
					}
		except Exception:
			pass

		# Fraud confidence bar
		try:
			if isinstance(fraud_data, dict):
				conf = fraud_data.get("confidence_score")
				is_fraud = fraud_data.get("is_fraudulent")
				if isinstance(conf, (int, float)):
					charts["fraud"] = {
						"$schema": "https://vega.github.io/schema/vega-lite/v5.json",
						"description": "Fraud confidence",
						"data": {"values": [{"metric": "Fraud Confidence", "value": float(conf)}]},
						"mark": "bar",
						"encoding": {
							"x": {"field": "metric", "type": "nominal", "title": ""},
							"y": {"field": "value", "type": "quantitative", "title": "Confidence"}
						}
					}
				if isinstance(is_fraud, bool):
					charts["fraud_flag"] = {"is_fraudulent": is_fraud}
		except Exception:
			pass

		# Export images (best effort)
		chart_images: Dict[str, str] = {}
		for name, spec in charts.items():
			uri = _spec_to_png_data_uri(spec)
			if uri:
				chart_images[name] = uri

		# Build Plotly figure JSONs for frontend rendering
		plotly_figs: Dict[str, Any] = {}
		# Projection line figure
		try:
			proj = charts.get("projection")
			if proj and isinstance(proj, dict):
				values = proj.get("data", {}).get("values", [])
				x_vals = [v.get("year") for v in values]
				y_vals = [v.get("balance") for v in values]
				if len(x_vals) >= 2 and len(y_vals) >= 2:
					plotly_figs["projection"] = {
						"data": [
							{"type": "scatter", "mode": "lines+markers", "x": x_vals, "y": y_vals, "name": "Balance"}
						],
						"layout": {
							"title": "Projected balance over time",
							"xaxis": {"title": "Year"},
							"yaxis": {"title": "Balance ($)"}
						}
					}
			
			# New comprehensive pension charts
			pension_progress = charts.get("pension_progress")
			if pension_progress and isinstance(pension_progress, dict):
				values = pension_progress.get("data", {}).get("values", [])
				categories = [v.get("category") for v in values]
				amounts = [v.get("amount") for v in values]
				if categories and amounts:
					plotly_figs["pension_progress"] = {
						"data": [
							{"type": "bar", "x": categories, "y": amounts, "name": "Amount"}
						],
						"layout": {
							"title": "Retirement Goal Progress",
							"xaxis": {"title": ""},
							"yaxis": {"title": "Amount ($)"}
						}
					}
			
			# Progress gauge
			progress_gauge = charts.get("progress_gauge")
			if progress_gauge and isinstance(progress_gauge, dict):
				values = progress_gauge.get("data", {}).get("values", [])
				if values and len(values) > 0:
					progress = values[0].get("progress", 0)
					plotly_figs["progress_gauge"] = {
						"data": [
							{"type": "indicator", "mode": "gauge+number+delta", "value": progress, "gauge": {"axis": {"range": [None, 100]}, "bar": {"color": "darkblue"}, "steps": [{"range": [0, 25], "color": "lightgray"}, {"range": [25, 50], "color": "yellow"}, {"range": [50, 100], "color": "green"}]}}
						],
						"layout": {
							"title": "Progress to Goal (%)",
							"width": 400,
							"height": 300
						}
					}
			
			# Retirement timeline
			retirement_timeline = charts.get("retirement_timeline")
			if retirement_timeline and isinstance(retirement_timeline, dict):
				values = retirement_timeline.get("data", {}).get("values", [])
				x_vals = [v.get("year") for v in values]
				y_vals = [v.get("balance") for v in values]
				types = [v.get("type", "Unknown") for v in values]
				if len(x_vals) >= 2 and len(y_vals) >= 2:
					plotly_figs["retirement_timeline"] = {
						"data": [
							{"type": "scatter", "mode": "lines+markers", "x": x_vals, "y": y_vals, "name": "Balance", "text": types, "hovertemplate": "Year: %{x}<br>Balance: $%{y:,.0f}<br>Type: %{text}<extra></extra>"}
						],
						"layout": {
							"title": "Years to Retirement",
							"xaxis": {"title": "Years"},
							"yaxis": {"title": "Balance ($)"}
						}
					}
			
			# Projection comparison chart
			projection_comparison = charts.get("projection_comparison")
			if projection_comparison and isinstance(projection_comparison, dict):
				values = projection_comparison.get("data", {}).get("values", [])
				types = [v.get("type") for v in values]
				amounts = [v.get("amount") for v in values]
				if types and amounts:
					plotly_figs["projection_comparison"] = {
						"data": [
							{"type": "bar", "x": types, "y": amounts, "name": "Projection Amount", "text": [f"${amount:,.0f}" for amount in amounts], "textposition": "auto"}
						],
						"layout": {
							"title": "Nominal vs Inflation-Adjusted Projection",
							"xaxis": {"title": ""},
							"yaxis": {"title": "Amount ($)"}
						}
					}
		except Exception:
			pass

		# Risk bar figure
		try:
			risk = charts.get("risk")
			if risk and isinstance(risk, dict):
				vals = risk.get("data", {}).get("values", [])
				if vals and isinstance(vals[0], dict) and "value" in vals[0]:
					plotly_figs["risk"] = {
						"data": [
							{"type": "bar", "x": ["Risk Score"], "y": [float(vals[0]["value"])], "name": "Risk Score"}
						],
						"layout": {"title": "Risk score"}
					}
		except Exception:
			pass

		# Fraud confidence bar figure
		try:
			fraud = charts.get("fraud")
			if fraud and isinstance(fraud, dict):
				vals = fraud.get("data", {}).get("values", [])
				if vals and isinstance(vals[0], dict) and "value" in vals[0]:
					plotly_figs["fraud"] = {
						"data": [
							{"type": "bar", "x": ["Fraud Confidence"], "y": [float(vals[0]["value"])], "name": "Fraud Confidence"}
						],
						"layout": {"title": "Fraud confidence"}
					}
		except Exception:
			pass

		messages = list(state["messages"])  # type: ignore
		messages.append(AIMessage(content="[CHART_SPECS] " + str(charts)))
		if chart_images:
			messages.append(AIMessage(content="[CHART_IMAGES] " + str(chart_images)))
		if plotly_figs:
			messages.append(AIMessage(content="[PLOTLY_FIGS] " + str(plotly_figs)))

		updates: Dict[str, Any] = {"messages": messages}
		if charts:
			updates["charts"] = charts
		if chart_images:
			updates["chart_images"] = chart_images
		if plotly_figs:
			updates["plotly_figs"] = plotly_figs
		return updates

	return visualizer_node