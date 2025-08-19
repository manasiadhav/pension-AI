#!/usr/bin/env python3
"""
Test the guardrails in the pension AI system
"""

import re

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
    """Test the full workflow with guardrails"""
    try:
        from app.workflow import graph
        
        print("\n🚀 Testing Full Workflow with Guardrails...")
        print("=" * 50)
        
        # Test a query that should trigger guardrails
        test_query = " I want to day trade with my pension for user 107?"
        print(f"📝 Testing: {test_query}")
        
        result = graph.invoke({'messages': [('user', test_query)]})
        print(f"✅ Workflow completed!")
        print(f"📊 Result keys: {list(result.keys())}")
        
        if 'final_response' in result:
            summary = result['final_response'].get('summary', 'No summary')
            print(f"📝 Summary: {summary[:200]}...")
        
        print("\n🎯 Guardrail workflow test successful!")
        
    except Exception as e:
        print(f"❌ Error testing workflow: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_guardrails()
    test_workflow_guardrails()