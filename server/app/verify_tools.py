#!/usr/bin/env python3
"""
Test the guardrails in the pension AI system
"""

import re
import json

def test_guardrails():
    """Test the content guardrails"""
    
    # Define blocked content patterns (same as in supervisor.py)
    blocked_patterns = {
        'religious': [
            r'\b(pray|prayer|god|jesus|allah|buddha|hindu|islam|christian|jewish|religious|spiritual|faith|blessing|divine|heaven|hell)\b',
            r'\b(amen|hallelujah|om|namaste|shalom|salaam)\b',
            r'\b(church|mosque|temple|synagogue|worship|meditation)\b'
        ],
        'political': [
            r'\b(democrat|republican|liberal|conservative|left|right|wing|party|election|vote|campaign|politician|senator|congress|president)\b',
            r'\b(government|administration|policy|legislation|bill|law|regulation)\b',
            r'\b(progressive|moderate|radical|extremist|activist|protest|rally)\b'
        ],
        'investment_strategy': [
            r'\b(buy|sell|hold|stock|shares|equity|market|timing|entry|exit|position|portfolio|allocation)\b',
            r'\b(day trading|swing trading|momentum|value|growth|dividend|yield)\b',
            r'\b(cryptocurrency|bitcoin|ethereum|blockchain|ico|token|coin)\b',
            r'\b(real estate|property|mortgage|loan|credit|debt|leverage)\b',
            r'\b(hedge fund|private equity|venture capital|startup|ipo|merger|acquisition)\b'
        ]
    }
    
    def validate_query_content(query: str) -> tuple[bool, str]:
        """Validate query content and return (is_valid, reason_if_invalid)"""
        
        # Check for blocked content
        for category, patterns in blocked_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query.lower()):
                    return False, f"Query contains {category} content which is not allowed"
        
        return True, ""
    
    # Test queries
    test_queries = [
        "What's the distribution of my pension fund for user 102.",
        "Should I pray for better pension returns?",
        "What's the political impact on my pension?",
        "Should I buy Bitcoin with my pension money?",
        "How does my pension grow over time?",
        "What's my risk profile?",
        "Is this transaction fraudulent?",
        "God bless my pension investments",
        "The government should regulate pensions more",
        "I want to day trade with my pension"
    ]
    
    print("🧪 Testing Content Guardrails...")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        is_valid, reason = validate_query_content(query)
        status = "✅ ALLOWED" if is_valid else "❌ BLOCKED"
        print(f"{i:2d}. {status}: {query}")
        if not is_valid:
            print(f"    Reason: {reason}")
    
    print("\n🎯 Guardrail Testing Summary:")
    print("- ✅ ALLOWED: Pension analysis, risk assessment, fraud detection")
    print("- ❌ BLOCKED: Religious content, political topics, investment strategies")
    print("- 🛡️ Protection: System will reject inappropriate queries early")

def test_workflow_guardrails():
    """Test the full workflow with guardrails and context system"""
    try:
        from app.workflow import graph
        from app.tools.tools import set_request_user_id, clear_request_user_id
        
        print("🚀 Testing Full Workflow with Guardrails...")
        print("=" * 50)
        
        # Test 1: Guardrail Test (should block political content)
        test_query = "What's the political impact on my pension for user 107?"
        print(f"📝 Test 1 - Guardrail Test: {test_query}")
        
        try:
            result = graph.invoke({'messages': [('user', test_query)]})
            print(f"✅ Workflow completed!")
            print(f"📊 Result keys: {list(result.keys())}")
            
            if 'final_response' in result:
                summary = result['final_response'].get('summary', 'No summary')
                print(f"📝 Summary: {summary[:200]}...")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "-" * 50)
        
        # Test 2: Request Context-Based User ID (should work with context)
        test_query = "What's my pension balance?"
        print(f"📝 Test 2 - Request Context-Based User ID: {test_query}")
        
        # Simulate frontend authentication - set user_id in request context
        set_request_user_id(102)  # This is what your FastAPI endpoint would do after JWT validation
        print(f"🔍 Context: Set user_id=102 in request context")
        
        try:
            result = graph.invoke({'messages': [('user', test_query)]})
            print(f"✅ Workflow completed!")
            print(f"📊 Result keys: {list(result.keys())}")
            
            if 'final_response' in result:
                summary = result['final_response'].get('summary', 'No summary')
                print(f"📝 Summary: {summary[:200]}...")
                
                # Check if charts were generated
                charts = result.get('charts', {})
                plotly_figs = result.get('plotly_figs', {})
                if charts or plotly_figs:
                    print(f"📊 Charts generated: {list(charts.keys())}")
                    print(f"📊 Plotly figures: {list(plotly_figs.keys())}")
                    
                    # Show actual JSON output for frontend
                    if plotly_figs:
                        print(f"\n🎨 ACTUAL JSON OUTPUT FOR FRONTEND:")
                        print(f"📊 Plotly Figures JSON:")
                        for chart_name, chart_data in plotly_figs.items():
                            print(f"\n📈 Chart: {chart_name}")
                            print(f"JSON Data: {json.dumps(chart_data, indent=2)}")
                    
                    if charts:
                        print(f"\n📊 Vega-Lite Charts JSON:")
                        for chart_name, chart_data in charts.items():
                            print(f"\n📈 Chart: {chart_name}")
                            print(f"JSON Data: {json.dumps(chart_data, indent=2)}")
                else:
                    print(f"📊 No charts generated")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            # Clean up context (FastAPI would do this automatically)
            clear_request_user_id()
            print(f"🔍 Context: Cleared user_id from request context")
        
        print("\n" + "-" * 50)
        
        # Test 3: No User Context (should fail gracefully)
        test_query = "Show me my risk profile"
        print(f"📝 Test 3 - No User Context: {test_query}")
        
        try:
            result = graph.invoke({'messages': [('user', test_query)]})
            print(f"✅ Workflow completed!")
            print(f"📊 Result keys: {list(result.keys())}")
            
            if 'final_response' in result:
                summary = result['final_response'].get('summary', 'No summary')
                print(f"📝 Summary: {summary[:200]}...")
                
                # Check if charts were generated
                charts = result.get('charts', {})
                plotly_figs = result.get('plotly_figs', {})
                if charts or plotly_figs:
                    print(f"📊 Charts generated: {list(charts.keys())}")
                    print(f"📊 Plotly figures: {list(plotly_figs.keys())}")
                    
                    # Show actual JSON output for frontend
                    if plotly_figs:
                        print(f"\n🎨 ACTUAL JSON OUTPUT FOR FRONTEND:")
                        print(f"📊 Plotly Figures JSON:")
                        for chart_name, chart_data in plotly_figs.items():
                            print(f"\n📈 Chart: {chart_name}")
                            print(f"JSON Data: {json.dumps(chart_data, indent=2)}")
                    
                    if charts:
                        print(f"\n📊 Vega-Lite Charts JSON:")
                        for chart_name, chart_data in charts.items():
                            print(f"\n📈 Chart: {chart_name}")
                            print(f"JSON Data: {json.dumps(chart_data, indent=2)}")
                else:
                    print(f"📊 No charts generated")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "-" * 50)
        
        # Test 4: Visualizer Agent Test (should generate charts)
        test_query = "Show me a chart of my risk profile"
        print(f"📝 Test 4 - Visualizer Agent Test: {test_query}")
        
        # Set user context for this test
        set_request_user_id(102)
        print(f"🔍 Context: Set user_id=102 in request context")
        
        try:
            result = graph.invoke({'messages': [('user', test_query)]})
            print(f"✅ Workflow completed!")
            print(f"📊 Result keys: {list(result.keys())}")
            
            if 'final_response' in result:
                summary = result['final_response'].get('summary', 'No summary')
                print(f"📝 Summary: {summary[:200]}...")
            
            # Check if charts were generated
            charts = result.get('charts', {})
            plotly_figs = result.get('plotly_figs', {})
            if charts or plotly_figs:
                print(f"📊 Charts generated: {list(charts.keys())}")
                print(f"📊 Plotly figures: {list(plotly_figs.keys())}")
                
                # Show actual JSON output for frontend
                if plotly_figs:
                    print(f"\n🎨 ACTUAL JSON OUTPUT FOR FRONTEND:")
                    print(f"📊 Plotly Figures JSON:")
                    for chart_name, chart_data in plotly_figs.items():
                        print(f"\n📈 Chart: {chart_name}")
                        print(f"JSON Data: {json.dumps(chart_data, indent=2)}")
                
                if charts:
                    print(f"\n📊 Vega-Lite Charts JSON:")
                    for chart_name, chart_data in charts.items():
                        print(f"\n📈 Chart: {chart_name}")
                        print(f"JSON Data: {json.dumps(chart_data, indent=2)}")
            else:
                print(f"📊 No charts generated")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            clear_request_user_id()
            print(f"🔍 Context: Cleared user_id from request context")
        
        print("\n" + "-" * 50)
        
    except Exception as e:
        print(f"❌ Error testing workflow: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_guardrails()
    test_workflow_guardrails()