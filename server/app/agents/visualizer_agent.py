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