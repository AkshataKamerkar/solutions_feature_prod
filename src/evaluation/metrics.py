"""
Enhanced Board Evaluation Metrics for marks 1-5
Detailed criteria for evaluating student answers across CBSE, ICSE, and State boards
"""

from typing import Dict, List, Any

class CBSEEvaluationMetrics:
    """Enhanced Board specific evaluation metrics for different mark categories"""
    
    EVALUATION_METRICS = {
        1: {
            "mark_value": 1,
            "answer_length": "30-40 words",
            "time_allocation": "1-2 minutes",
            "evaluation_criteria": [
                {
                    "criteria_id": "1.1",
                    "criteria_name": "Answer Accuracy",
                    "description": "Correctness of the answer/option selected",
                    "max_marks": 1.0,
                    "marking_scheme": {
                        "correct": 1.0,
                        "incorrect": 0.0
                    },
                    "evaluation_points": [
                        "Check if answer matches marking scheme/answer key",
                        "For MCQ: Verify correct option is clearly marked",
                        "For fill-in-blanks: Exact or equivalent answer accepted",
                        "For True/False: Clear indication with reason if asked",
                        "For one-word answers: Spelling variations accepted if phonetically correct"
                    ],
                    "board_specific": {
                        "CBSE": "Accept NCERT terminology only",
                        "ICSE": "Technical terms must be precise",
                        "SSC": "Local language equivalents accepted"
                    }
                },
                {
                    "criteria_id": "1.2",
                    "criteria_name": "Reasoning Quality",
                    "description": "If reasoning required, evaluate logical support",
                    "max_marks": 0.0,
                    "marking_scheme": {
                        "logical_reasoning": "Bonus 0.5 if exceptionally clear",
                        "no_reasoning": "No penalty if not asked"
                    },
                    "evaluation_points": [
                        "Check if reasoning directly supports the answer",
                        "One-line explanation sufficient",
                        "Scientific/mathematical logic preferred"
                    ],
                    "optional": True,
                    "board_specific": {
                        "CBSE": "NCERT-based reasoning expected",
                        "ICSE": "Advanced reasoning appreciated",
                        "SSC": "Simple practical reasoning accepted"
                    }
                }
            ],
            "general_instructions": "For 1-mark questions, focus on accuracy. No penalty for minor spelling errors unless meaning changes.",
            "common_mistakes": [
                "Writing lengthy answers (time waste)",
                "Not attempting due to doubt",
                "Changing answer multiple times"
            ]
        },
        
        2: {
            "mark_value": 2,
            "answer_length": "60-80 words",
            "time_allocation": "3-4 minutes",
            "evaluation_criteria": [
                {
                    "criteria_id": "2.1",
                    "criteria_name": "Concept Understanding",
                    "description": "Understanding of basic concept/formula/definition",
                    "max_marks": 1.0,
                    "marking_scheme": {
                        "complete_understanding": 1.0,
                        "partial_understanding": 0.5,
                        "incorrect_concept": 0.0
                    },
                    "evaluation_points": [
                        "Definition/formula correctly stated",
                        "Key terms properly used",
                        "Concept clearly explained",
                        "Proper mathematical/scientific notation"
                    ],
                    "board_specific": {
                        "CBSE": "NCERT definitions mandatory",
                        "ICSE": "Extended definitions expected",
                        "SSC": "Simple language definitions accepted"
                    }
                },
                {
                    "criteria_id": "2.2",
                    "criteria_name": "Application/Example",
                    "description": "Practical application or example provided",
                    "max_marks": 0.5,
                    "marking_scheme": {
                        "relevant_example": 0.5,
                        "partially_relevant": 0.25,
                        "no_example": 0.0
                    },
                    "evaluation_points": [
                        "Example directly relates to concept",
                        "Real-life application shown",
                        "Numerical example with correct calculation",
                        "Diagram/illustration if helpful"
                    ],
                    "board_specific": {
                        "CBSE": "Indian context examples preferred",
                        "ICSE": "Global examples accepted",
                        "SSC": "Local/regional examples encouraged"
                    }
                },
                {
                    "criteria_id": "2.3",
                    "criteria_name": "Answer Completeness",
                    "description": "All parts of question addressed",
                    "max_marks": 0.5,
                    "marking_scheme": {
                        "fully_complete": 0.5,
                        "partially_complete": 0.25,
                        "incomplete": 0.0
                    },
                    "evaluation_points": [
                        "Both parts answered if two-part question",
                        "Cause and effect both explained",
                        "Advantages and disadvantages both listed",
                        "Compare and contrast both aspects covered"
                    ]
                },
                {
                    "criteria_id": "2.4",
                    "criteria_name": "Presentation",
                    "description": "Clarity and organization",
                    "max_marks": 0.0,
                    "marking_scheme": {
                        "poor_presentation": -0.25,
                        "illegible_writing": -0.5
                    },
                    "evaluation_points": [
                        "Readable handwriting",
                        "Proper spacing and margins",
                        "Key terms underlined",
                        "Logical flow of content"
                    ],
                    "deduction_only": True
                }
            ],
            "general_instructions": "For 2-mark questions, balance between concept and application. Both components essential.",
            "answer_format": "Definition/Formula → Example/Application → Connection to question",
            "common_mistakes": [
                "Only writing definition without example",
                "Too lengthy explanation",
                "Missing units in calculations",
                "Not reading both parts of question"
            ]
        },
        
        3: {
            "mark_value": 3,
            "answer_length": "100-150 words",
            "time_allocation": "5-6 minutes",
            "evaluation_criteria": [
                {
                    "criteria_id": "3.1",
                    "criteria_name": "Introduction/Concept",
                    "description": "Clear introduction and concept identification",
                    "max_marks": 0.5,
                    "marking_scheme": {
                        "clear_introduction": 0.5,
                        "vague_introduction": 0.25,
                        "no_introduction": 0.0
                    },
                    "evaluation_points": [
                        "Topic clearly introduced",
                        "Given information stated",
                        "Required to find mentioned",
                        "Relevant theorem/principle identified"
                    ],
                    "board_specific": {
                        "CBSE": "Start with textbook definition",
                        "ICSE": "Historical context appreciated",
                        "SSC": "Simple introduction sufficient"
                    }
                },
                {
                    "criteria_id": "3.2",
                    "criteria_name": "Explanation/Working",
                    "description": "Main body with detailed explanation or working",
                    "max_marks": 1.5,
                    "marking_scheme": {
                        "complete_explanation": 1.5,
                        "mostly_correct": 1.0,
                        "partial_explanation": 0.5,
                        "incorrect_approach": 0.0
                    },
                    "evaluation_points": [
                        "Step-by-step explanation provided",
                        "All formulas clearly stated",
                        "Calculations shown in detail",
                        "Logical progression maintained",
                        "Scientific method followed"
                    ],
                                        "board_specific": {
                        "CBSE": "Follow NCERT solved examples pattern",
                        "ICSE": "Multiple methods shown if possible",
                        "SSC": "Step-wise simple approach"
                    }
                },
                {
                    "criteria_id": "3.3",
                    "criteria_name": "Diagram/Example",
                    "description": "Supporting diagram or worked example",
                    "max_marks": 0.5,
                    "marking_scheme": {
                        "accurate_diagram": 0.5,
                        "partially_correct": 0.25,
                        "no_diagram_when_needed": 0.0
                    },
                    "evaluation_points": [
                        "Diagram neat and labeled",
                        "Correct proportions maintained",
                        "All parts clearly marked",
                        "Example calculations complete",
                        "Units mentioned throughout"
                    ],
                    "board_specific": {
                        "CBSE": "NCERT diagram style required",
                        "ICSE": "Detailed artistic diagrams",
                        "SSC": "Basic sketches accepted"
                    }
                },
                {
                    "criteria_id": "3.4",
                    "criteria_name": "Conclusion",
                    "description": "Clear conclusion with final answer",
                    "max_marks": 0.5,
                    "marking_scheme": {
                        "clear_conclusion": 0.5,
                        "vague_conclusion": 0.25,
                        "no_conclusion": 0.0
                    },
                    "evaluation_points": [
                        "Final answer clearly stated",
                        "Units mentioned correctly",
                        "Answer relates to question asked",
                        "Key findings summarized",
                        "Real-world relevance mentioned"
                    ]
                },
                {
                    "criteria_id": "3.5",
                    "criteria_name": "Overall Quality",
                    "description": "Presentation and completeness",
                    "max_marks": 0.0,
                    "marking_scheme": {
                        "exceptional_presentation": 0.5,
                        "poor_presentation": -0.5
                    },
                    "evaluation_points": [
                        "Neat and organized layout",
                        "Proper use of space",
                        "Systematic approach",
                        "Time management evident"
                    ],
                    "bonus_deduction": True
                }
            ],
            "general_instructions": "For 3-mark questions, expect structured answers with clear beginning, middle, and end.",
            "answer_format": "Introduction → Main Explanation/Working → Diagram/Example → Conclusion",
            "time_management": "Spend 1 minute planning, 4 minutes writing, 1 minute reviewing",
            "common_mistakes": [
                "Missing diagram when required",
                "Incomplete calculations",
                "No clear conclusion",
                "Poor time management",
                "Forgetting units"
            ]
        },
        
        4: {
            "mark_value": 4,
            "answer_length": "200-250 words",
            "time_allocation": "7-8 minutes",
            "evaluation_criteria": [
                {
                    "criteria_id": "4.1",
                    "criteria_name": "Problem Analysis",
                    "description": "Understanding and analysis of the problem",
                    "max_marks": 1.0,
                    "marking_scheme": {
                        "comprehensive_analysis": 1.0,
                        "good_analysis": 0.75,
                        "basic_analysis": 0.5,
                        "poor_analysis": 0.25,
                        "no_analysis": 0.0
                    },
                    "evaluation_points": [
                        "All given information identified",
                        "Hidden information deduced",
                        "Variables clearly defined",
                        "Assumptions stated if any",
                        "Problem broken into parts",
                        "Relationships identified"
                    ],
                    "board_specific": {
                        "CBSE": "Use standard NCERT problem-solving approach",
                        "ICSE": "Show analytical thinking process",
                        "SSC": "Simple breakdown sufficient"
                    }
                },
                {
                    "criteria_id": "4.2",
                    "criteria_name": "Conceptual Application",
                    "description": "Application of multiple concepts/formulas",
                    "max_marks": 1.5,
                    "marking_scheme": {
                        "all_concepts_correct": 1.5,
                        "mostly_correct": 1.0,
                        "some_concepts_correct": 0.75,
                        "few_concepts_correct": 0.5,
                        "major_conceptual_errors": 0.0
                    },
                    "evaluation_points": [
                        "Relevant formulas identified",
                        "Correct theoretical basis",
                        "Multiple concepts integrated",
                        "Proper justification given",
                        "Alternative approaches considered",
                        "Cross-chapter connections made"
                    ],
                    "board_specific": {
                        "CBSE": "Stick to NCERT concepts only",
                        "ICSE": "Advanced concepts welcome",
                        "SSC": "Basic concepts sufficient"
                    }
                },
                {
                    "criteria_id": "4.3",
                    "criteria_name": "Solution Process",
                    "description": "Step-by-step solution with all working",
                    "max_marks": 1.0,
                    "marking_scheme": {
                        "perfect_solution": 1.0,
                        "minor_errors": 0.75,
                        "some_correct_steps": 0.5,
                        "major_errors": 0.25,
                        "incorrect_method": 0.0
                    },
                    "evaluation_points": [
                        "Logical sequence maintained",
                        "All steps clearly shown",
                        "Intermediate results checked",
                        "Units carried throughout",
                        "Calculations accurate",
                        "Method clearly explained"
                    ],
                    "subject_specific": {
                        "Mathematics": "Show all algebraic steps",
                        "Science": "Include all conversions",
                        "Social Science": "Chronological/logical order"
                    }
                },
                {
                    "criteria_id": "4.4",
                    "criteria_name": "Result Interpretation",
                    "description": "Final answer with proper interpretation",
                    "max_marks": 0.5,
                    "marking_scheme": {
                        "excellent_interpretation": 0.5,
                        "good_interpretation": 0.35,
                        "basic_interpretation": 0.25,
                        "no_interpretation": 0.0
                    },
                    "evaluation_points": [
                        "Answer addresses question completely",
                        "Real-world significance explained",
                        "Limitations mentioned if any",
                        "Answer verified/checked",
                        "Alternative solutions discussed",
                        "Connection to larger concept"
                    ]
                }
            ],
            "general_instructions": "For 4-mark questions, demonstrate comprehensive understanding with multiple concepts.",
            "answer_format": "Analysis → Theory/Concepts → Detailed Solution → Interpretation → Verification",
            "case_study_specific": "Read all parts, identify common theme, use data provided, answer in context",
            "common_mistakes": [
                "Not reading case study completely",
                "Missing data from graphs/tables",
                "Incomplete interpretation",
                "Not showing all work",
                "Forgetting real-world context"
            ]
        },
        
        5: {
            "mark_value": 5,
            "answer_length": "300-350 words",
            "time_allocation": "10-12 minutes",
            "evaluation_criteria": [
                {
                    "criteria_id": "5.1",
                    "criteria_name": "Complete Understanding",
                    "description": "Comprehensive understanding of all aspects",
                    "max_marks": 1.0,
                    "marking_scheme": {
                        "excellent_understanding": 1.0,
                        "good_understanding": 0.75,
                        "basic_understanding": 0.5,
                        "poor_understanding": 0.25,
                        "no_understanding": 0.0
                    },
                    "evaluation_points": [
                        "All concepts correctly identified",
                        "Relationships clearly shown",
                        "Depth of knowledge evident",
                        "Cross-connections made",
                        "Advanced applications shown",
                        "Historical/future context given"
                    ],
                    "board_specific": {
                        "CBSE": "NCERT + practical applications",
                        "ICSE": "In-depth analytical approach",
                        "SSC": "Comprehensive basic coverage"
                    }
                },
                {
                    "criteria_id": "5.2",
                    "criteria_name": "Theoretical Foundation",
                    "description": "Strong theoretical basis with formulas/principles",
                    "max_marks": 1.0,
                    "marking_scheme": {
                        "complete_theory": 1.0,
                        "mostly_complete": 0.75,
                        "partial_theory": 0.5,
                        "weak_theory": 0.25,
                        "incorrect_theory": 0.0
                    },
                    "evaluation_points": [
                        "All formulas correctly stated",
                        "Derivations shown if needed",
                        "Principles properly explained",
                        "Assumptions clearly listed",
                        "Limitations acknowledged",
                        "Theoretical basis justified"
                    ]
                },
                {
                    "criteria_id": "5.3",
                    "criteria_name": "Detailed Working",
                    "description": "Complete step-by-step solution",
                    "max_marks": 2.0,
                    "marking_scheme": {
                        "perfect_working": 2.0,
                        "minor_calculation_errors": 1.5,
                        "some_steps_missing": 1.0,
                        "major_gaps": 0.5,
                        "incorrect_approach": 0.0
                    },
                    "evaluation_points": [
                        "Every step clearly shown",
                        "Rough work included",
                        "Alternative methods shown",
                        "Verification steps included",
                        "Error analysis if applicable",
                        "Systematic presentation"
                    ],
                    "marking_breakdown": {
                        "Method selection": 0.5,
                        "Step-wise execution": 1.0,
                        "Accuracy": 0.5
                    }
                },
                {
                    "criteria_id": "5.4",
                    "criteria_name": "Visual Representation",
                    "description": "Diagrams, graphs, or tables as required",
                    "max_marks": 0.5,
                    "marking_scheme": {
                        "excellent_visuals": 0.5,
                        "good_visuals": 0.35,
                        "basic_visuals": 0.25,
                        "poor_visuals": 0.1,
                        "missing_required_visuals": 0.0
                    },
                    "evaluation_points": [
                        "Diagrams neat and labeled",
                        "Graphs with proper scale",
                        "Tables well-organized",
                        "Color coding if helpful",
                        "Legends/keys included",
                        "Visual aids enhance answer"
                    ],
                    "subject_requirements": {
                        "Physics": "Circuit/ray diagrams mandatory",
                        "Chemistry": "Structural formulas required",
                        "Biology": "Labeled diagrams essential",
                        "Geography": "Maps/charts needed"
                    }
                },
                {
                    "criteria_id": "5.5",
                    "criteria_name": "Conclusion & Application",
                    "description": "Strong conclusion with real-world applications",
                    "max_marks": 0.5,
                    "marking_scheme": {
                        "outstanding_conclusion": 0.5,
                        "good_conclusion": 0.35,
                        "basic_conclusion": 0.25,
                        "weak_conclusion": 0.1,
                        "no_conclusion": 0.0
                    },
                    "evaluation_points": [
                        "Summary of key findings",
                        "Real-world applications",
                        "Future implications",
                        "Environmental/social impact",
                        "Personal reflection",
                        "Connection to other topics"
                    ]
                }
            ],
            "general_instructions": "For 5-mark questions, provide comprehensive answers demonstrating mastery of topic.",
            "answer_format": "Introduction → Theory → Detailed Working → Diagrams → Applications → Conclusion",
            "time_distribution": {
                "Planning": "2 minutes",
                "Writing": "8 minutes",
                "Diagrams": "1 minute",
                "Review": "1 minute"
            },
            "pro_tips": [
                "Start with rough work/plan",
                "Allocate space for diagrams",
                "Keep 20% space for additions",
                "Review calculations twice",
                "End with strong conclusion"
            ],
            "common_mistakes": [
                "Poor time management",
                "Missing diagrams",
                "Incomplete solutions",
                "No verification",
                "Weak conclusion",
                "Exceeding word limit"
            ]
        }
    }
    
    @classmethod
    def get_evaluation_metrics(cls, marks: int) -> Dict[str, Any]:
        """
        Get evaluation metrics for specific mark value
        
        Args:
            marks: The mark value (1-5)
                    
        Returns:
            Dictionary containing evaluation criteria for the specified marks
        """
        if marks not in cls.EVALUATION_METRICS:
            raise ValueError(f"Invalid marks value: {marks}. Must be between 1 and 5.")
        
        return cls.EVALUATION_METRICS[marks]
    
    @classmethod
    def get_evaluation_prompt(cls, marks: int, board: str = "CBSE") -> str:
        """
        Generate a comprehensive evaluation prompt for the given marks
        
        Args:
            marks: The mark value (1-5)
            board: Board type (CBSE/ICSE/SSC)
            
        Returns:
            Formatted evaluation prompt string
        """
        metrics = cls.get_evaluation_metrics(marks)
        
        # Board-specific headers
        board_headers = {
            "CBSE": "📘 CBSE BOARD EVALUATION STANDARDS",
            "ICSE": "📗 ICSE BOARD EVALUATION STANDARDS",
            "SSC": "📙 STATE BOARD EVALUATION STANDARDS"
        }
        
        header = board_headers.get(board.upper(), board_headers["CBSE"])
        
        prompt_parts = [
            f"{header}",
            f"\n🎯 Evaluating {marks}-mark answer based on {board} board standards.",
            f"\n📋 General Instructions: {metrics['general_instructions']}",
            f"\n⏱️ Expected Time: {metrics.get('time_allocation', 'As per marks')}",
            f"\n📝 Expected Length: {metrics.get('answer_length', 'As per marks')}",
            "\n\n📊 DETAILED EVALUATION CRITERIA:"
        ]
        
        # Add answer format if available
        if 'answer_format' in metrics:
            prompt_parts.append(f"\n✍️ Expected Answer Format: {metrics['answer_format']}")
        
        # Add evaluation criteria
        for criterion in metrics['evaluation_criteria']:
            if criterion.get('internal_use_only', False):
                continue
                
            prompt_parts.append(f"\n\n{criterion['criteria_id']}. {criterion['criteria_name']} (Max: {criterion['max_marks']} marks)")
            prompt_parts.append(f"   📌 {criterion['description']}")
            
            # Add marking scheme details
            if 'marking_scheme' in criterion:
                prompt_parts.append("\n   📊 Marking Scheme:")
                for condition, marks_value in criterion['marking_scheme'].items():
                    if isinstance(marks_value, (int, float)):
                        prompt_parts.append(f"     • {condition.replace('_', ' ').title()}: {marks_value} marks")
                    else:
                        prompt_parts.append(f"     • {condition.replace('_', ' ').title()}: {marks_value}")
            
            # Add evaluation points
            if 'evaluation_points' in criterion:
                prompt_parts.append("\n   ✓ Check Points:")
                for point in criterion['evaluation_points']:
                    prompt_parts.append(f"     - {point}")
            
            # Add board-specific notes
            if 'board_specific' in criterion and board.upper() in criterion['board_specific']:
                prompt_parts.append(f"\n   🎓 {board} Specific: {criterion['board_specific'][board.upper()]}")
            
            # Add subject-specific notes
            if 'subject_specific' in criterion:
                prompt_parts.append("\n   📚 Subject Requirements:")
                for subject, requirement in criterion['subject_specific'].items():
                    prompt_parts.append(f"     • {subject}: {requirement}")
        
        # Add common mistakes section
        if 'common_mistakes' in metrics:
            prompt_parts.append("\n\n⚠️ COMMON MISTAKES TO WATCH FOR:")
            for mistake in metrics['common_mistakes']:
                prompt_parts.append(f"   ❌ {mistake}")
        
        # Add pro tips if available
        if 'pro_tips' in metrics:
            prompt_parts.append("\n\n💡 PRO TIPS FOR STUDENTS:")
            for tip in metrics['pro_tips']:
                prompt_parts.append(f"   ✅ {tip}")
        
        # Add case study specific guidance
        if 'case_study_specific' in metrics:
            prompt_parts.append(f"\n\n📖 CASE STUDY GUIDANCE: {metrics['case_study_specific']}")
        
        # Add time management
        if 'time_distribution' in metrics:
            prompt_parts.append("\n\n⏰ TIME MANAGEMENT:")
            for phase, time in metrics['time_distribution'].items():
                prompt_parts.append(f"   • {phase}: {time}")
        
        prompt_parts.append("\n\n🎯 EVALUATION APPROACH:")
        prompt_parts.append("   1. Read complete answer first")
        prompt_parts.append("   2. Check against each criterion")
        prompt_parts.append("   3. Award marks step-wise")
        prompt_parts.append("   4. Consider alternative methods")
        prompt_parts.append("   5. Be lenient for 10th grade level")
        prompt_parts.append("\n\n📝 Provide detailed evaluation addressing each criterion with specific examples from the answer.")
        
        return "\n".join(prompt_parts)
    
    @classmethod
    def calculate_total_marks(cls, marks: int, criterion_scores: Dict[str, float], 
                            board: str = "CBSE") -> Dict[str, Any]:
        """
        Calculate total marks based on individual criterion scores
        
        Args:
            marks: The mark value (1-5)
            criterion_scores: Dictionary mapping criteria_id to scores
            board: Board type for specific adjustments
            
        Returns:
            Dictionary with total marks and breakdown
        """
        metrics = cls.get_evaluation_metrics(marks)
        total = 0.0
        breakdown = {}
        deductions = 0.0
        bonuses = 0.0
        
        for criterion in metrics['evaluation_criteria']:
            criteria_id = criterion['criteria_id']
            if criteria_id in criterion_scores:
                score = criterion_scores[criteria_id]
                
                # Handle deductions
                if criterion.get('deduction_only', False):
                    deductions += abs(score)
                    breakdown[f"{criteria_id}_deduction"] = score
                # Handle bonuses
                elif criterion.get('bonus_deduction', False):
                    if score > 0:
                        bonuses += score
                        breakdown[f"{criteria_id}_bonus"] = score
                    else:
                        deductions += abs(score)
                        breakdown[f"{criteria_id}_deduction"] = score
                else:
                    # Regular scoring
                    max_marks = criterion['max_marks']
                    actual_score = min(score, max_marks)
                    total += actual_score
                    breakdown[criteria_id] = actual_score
        
        # Apply board-specific adjustments
        board_adjustments = {
            "CBSE": {"leniency": 0.05, "max_bonus": 0.5},
            "ICSE": {"leniency": 0.0, "max_bonus": 0.25},
            "SSC": {"leniency": 0.1, "max_bonus": 0.75}
        }
        
        adjustment = board_adjustments.get(board.upper(), board_adjustments["CBSE"])
        
        # Apply leniency
        total = total * (1 + adjustment["leniency"])
        
        # Apply bonuses with cap
        total += min(bonuses, adjustment["max_bonus"])
        
        # Apply deductions
        total -= deductions
        
        # Ensure total doesn't exceed maximum possible marks
        final_total = max(0, min(total, marks))
        
        # Calculate percentage
        percentage = (final_total / marks) * 100
        
        # Determine grade
        grade = cls._calculate_grade(percentage)
        
        return {
            "total_marks": round(final_total, 2),
            "maximum_marks": marks,
            "percentage": round(percentage, 1),
            "grade": grade,
            "breakdown": breakdown,
            "deductions": round(deductions, 2),
            "bonuses": round(bonuses, 2),
            "board_adjustment": adjustment["leniency"]
        }
    
    @classmethod
    def _calculate_grade(cls, percentage: float) -> str:
        """Calculate grade based on percentage"""
        if percentage >= 90:
            return "A+ (Outstanding)"
        elif percentage >= 80:
            return "A (Excellent)"
        elif percentage >= 70:
            return "B+ (Very Good)"
        elif percentage >= 60:
            return "B (Good)"
        elif percentage >= 50:
            return "C+ (Above Average)"
        elif percentage >= 40:
            return "C (Average)"
        elif percentage >= 33:
            return "D (Pass)"
        else:
            return "E (Needs Improvement)"
    
    @classmethod
    def get_improvement_suggestions(cls, marks: int, criterion_scores: Dict[str, float], 
                                  board: str = "CBSE") -> List[str]:
        """
        Generate specific improvement suggestions based on performance
        
        Args:
            marks: The mark value (1-5)
            criterion_scores: Dictionary mapping criteria_id to scores
            board: Board type
            
        Returns:
            List of improvement suggestions
        """
        metrics = cls.get_evaluation_metrics(marks)
        suggestions = []
        
        for criterion in metrics['evaluation_criteria']:
            criteria_id = criterion['criteria_id']
            max_marks = criterion['max_marks']
            
            if criteria_id in criterion_scores and max_marks > 0:
                score = criterion_scores[criteria_id]
                percentage_scored = (score / max_marks) * 100 if max_marks > 0 else 0
                
                if percentage_scored < 50:
                    # Poor performance - needs significant improvement
                    suggestions.append(f"🔴 {criterion['criteria_name']}: Focus on {criterion['description'].lower()}")
                    
                    # Add specific tips based on criterion
                    if 'evaluation_points' in criterion and len(criterion['evaluation_points']) > 0:
                        suggestions.append(f"   Tip: {criterion['evaluation_points'][0]}")
                        
                elif percentage_scored < 75:
                    # Average performance - can improve
                    suggestions.append(f"🟡 {criterion['criteria_name']}: Practice more on this aspect")
                    
                elif percentage_scored < 100:
                    # Good performance - minor improvements
                    suggestions.append(f"🟢 {criterion['criteria_name']}: Almost perfect, minor refinements needed")
        
        # Add general suggestions based on marks
        general_suggestions = {
            1: ["Practice MCQs daily", "Focus on key terms", "Read questions carefully"],
            2: ["Balance theory and examples", "Keep answers concise", "Use proper format"],
            3: ["Include diagrams wherever possible", "Structure answers clearly", "Practice time management"],
            4: ["Analyze case studies thoroughly", "Show all working", "Connect concepts"],
            5: ["Plan before writing", "Include alternative methods", "Strong conclusions essential"]
        }
        
        if marks in general_suggestions:
            suggestions.extend([f"💡 {sug}" for sug in general_suggestions[marks]])
        
        # Board-specific suggestions
        board_suggestions = {
            "CBSE": "📚 Refer to NCERT solved examples and previous year papers",
            "ICSE": "📚 Practice from multiple reference books and focus on depth",
            "SSC": "📚 Focus on state board textbook and local examples"
        }
        
        if board.upper() in board_suggestions:
            suggestions.append(board_suggestions[board.upper()])
        
        return suggestions
    
    @classmethod
    def generate_model_answer_outline(cls, marks: int, board: str = "CBSE") -> str:
        """
        Generate a model answer outline for the given marks
        
        Args:
            marks: The mark value (1-5)
            board: Board type
            
        Returns:
            Model answer outline string
        """
        metrics = cls.get_evaluation_metrics(marks)
        
        outline_parts = [f"📝 MODEL ANSWER OUTLINE FOR {marks} MARKS ({board}):\n"]
        
        # Add format
        if 'answer_format' in metrics:
            outline_parts.append(f"FORMAT: {metrics['answer_format']}\n")
        
        # Add length and time
        outline_parts.append(f"LENGTH: {metrics.get('answer_length', 'As appropriate')}")
        outline_parts.append(f"TIME: {metrics.get('time_allocation', 'As per marks')}\n")
        
        # Add structure
        outline_parts.append("STRUCTURE:")
        
        # Generate structure based on criteria
        for i, criterion in enumerate(metrics['evaluation_criteria'], 1):
            if not criterion.get('deduction_only', False) and not criterion.get('internal_use_only', False):
                marks_worth = criterion['max_marks']
                if marks_worth > 0:
                    outline_parts.append(f"{i}. {criterion['criteria_name']} ({marks_worth} marks)")
                    if 'evaluation_points' in criterion:
                        for point in criterion['evaluation_points'][:2]:  # Show first 2 points
                            outline_parts.append(f"   - {point}")
        
        # Add tips
        if 'pro_tips' in metrics:
            outline_parts.append("\nKEY TIPS:")
            for tip in metrics['pro_tips'][:3]:  # Show first 3 tips
                outline_parts.append(f"• {tip}")
        
                return "\n".join(outline_parts)
    
    @classmethod
    def get_subject_specific_guidelines(cls, subject: str, marks: int) -> Dict[str, Any]:
        """
        Get subject-specific evaluation guidelines
        
        Args:
            subject: Subject name
            marks: Mark value (1-5)
            
        Returns:
            Dictionary of subject-specific guidelines
        """
        subject_guidelines = {
            "Mathematics": {
                "formula_requirement": True,
                "step_marking": True,
                "diagram_types": ["Geometric figures", "Graphs", "Number lines"],
                "common_errors": ["Calculation mistakes", "Missing units", "Wrong formula"],
                "presentation": "Show all steps, box final answers",
                "verification": "Check by substitution or alternative method"
            },
            "Science": {
                "formula_requirement": True,
                "step_marking": True,
                "diagram_types": ["Ray diagrams", "Circuit diagrams", "Biological diagrams"],
                "common_errors": ["Wrong units", "Missing labels", "Incorrect conversions"],
                "presentation": "Use SI units, label all diagrams",
                "experiments": "Include precautions and sources of error"
            },
            "Social Science": {
                "formula_requirement": False,
                "step_marking": False,
                "diagram_types": ["Maps", "Flow charts", "Timeline diagrams"],
                "common_errors": ["Wrong dates", "Factual errors", "Missing examples"],
                "presentation": "Use headings, point-wise answers",
                "maps": "Mark and label clearly"
            },
            "English": {
                "formula_requirement": False,
                "step_marking": False,
                "diagram_types": [],
                "common_errors": ["Grammar mistakes", "Spelling errors", "Off-topic"],
                "presentation": "Paragraph format, clear handwriting",
                "quotes": "Use quotation marks, mention source"
            }
        }
        
        # Get base guidelines
        base_guidelines = subject_guidelines.get(subject, {
            "formula_requirement": False,
            "step_marking": True,
            "diagram_types": [],
            "common_errors": ["Incomplete answers", "Poor presentation"],
            "presentation": "Neat and organized"
        })
        
        # Add marks-specific modifications
        marks_specific = {
            1: {"focus": "Accuracy only", "detail_level": "Minimal"},
            2: {"focus": "Concept + Example", "detail_level": "Basic"},
            3: {"focus": "Complete explanation", "detail_level": "Moderate"},
            4: {"focus": "Analysis + Application", "detail_level": "Detailed"},
            5: {"focus": "Comprehensive mastery", "detail_level": "Extensive"}
        }
        
        base_guidelines.update(marks_specific.get(marks, {}))
        
        return base_guidelines
    
    @classmethod
    def get_time_management_guide(cls, total_marks: int, question_distribution: Dict[int, int]) -> Dict[str, Any]:
        """
        Generate time management guide for exam
        
        Args:
            total_marks: Total marks in exam
            question_distribution: Dict mapping mark value to number of questions
            
        Returns:
            Time management guide
        """
        total_time = 180  # 3 hours in minutes
        reading_time = 15
        buffer_time = 15
        available_time = total_time - reading_time - buffer_time
        
        time_allocation = {}
        total_questions = 0
        marks_accounted = 0
        
        for marks, count in question_distribution.items():
            if marks in cls.EVALUATION_METRICS:
                time_per_question = cls.EVALUATION_METRICS[marks].get('time_allocation', f"{marks * 2} minutes")
                # Extract numeric value
                if isinstance(time_per_question, str):
                    time_value = int(time_per_question.split('-')[1].split()[0])
                else:
                    time_value = marks * 2
                
                total_time_for_type = time_value * count
                time_allocation[f"{marks}_mark_questions"] = {
                    "count": count,
                    "time_per_question": time_value,
                    "total_time": total_time_for_type,
                    "percentage": round((total_time_for_type / available_time) * 100, 1)
                }
                
                total_questions += count
                marks_accounted += marks * count
        
        return {
            "exam_duration": f"{total_time} minutes",
            "reading_time": f"{reading_time} minutes",
            "writing_time": f"{available_time} minutes",
            "buffer_time": f"{buffer_time} minutes (revision)",
            "total_marks": total_marks,
            "total_questions": total_questions,
            "time_allocation": time_allocation,
            "tips": [
                "Read all questions during reading time",
                "Start with questions you know best",
                "Allocate time proportional to marks",
                "Keep last 15 minutes for revision",
                "Don't spend too much time on one question"
            ]
        }
    
    @classmethod
    def get_board_specific_tips(cls, board: str) -> List[str]:
        """
        Get board-specific exam tips
        
        Args:
            board: Board name
            
        Returns:
            List of board-specific tips
        """
        board_tips = {
            "CBSE": [
                "📚 Stick to NCERT textbook content",
                "✍️ Use exact NCERT terminology",
                "📊 Include value-based points where relevant",
                "🎯 Focus on step-wise marking",
                "📝 Alternative methods are accepted",
                "💡 Practical examples from Indian context",
                "🔍 Previous year patterns repeat",
                "⏰ Time management is crucial"
            ],
            "ICSE": [
                "📚 Go beyond textbook for depth",
                "✍️ Use precise technical language",
                "📊 Include multiple examples and applications",
                "🎯 Quality over quantity approach",
                "📝 Show comprehensive understanding",
                "💡 International examples welcomed",
                "🔍 Analytical questions common",
                "⏰ Plan answers before writing"
            ],
            "SSC": [
                "📚 Focus on state board textbook",
                "✍️ Simple language is acceptable",
                "📊 Use local/regional examples",
                "🎯 Point-wise answers score well",
                "📝 Basic understanding sufficient",
                "💡 Practical knowledge valued",
                "🔍 Direct questions common",
                "⏰ Attempt all questions"
            ]
        }
        
        return board_tips.get(board.upper(), [
            "📚 Study from prescribed textbook",
            "✍️ Write clearly and neatly",
            "📊 Include examples where asked",
            "🎯 Follow marking scheme",
            "📝 Answer what is asked",
            "💡 Practice previous papers",
            "🔍 Read questions carefully",
            "⏰ Manage time wisely"
        ])
    
    @classmethod
    def evaluate_answer_quality(cls, answer: str, marks: int) -> Dict[str, Any]:
        """
        Quick quality check of answer based on basic metrics
        
        Args:
            answer: Student's answer text
            marks: Expected marks
            
        Returns:
            Quality assessment dictionary
        """
        metrics = cls.get_evaluation_metrics(marks)
        expected_length = metrics.get('answer_length', '50-100 words')
        
        # Extract expected word count range
        if '-' in expected_length:
            min_words = int(expected_length.split('-')[0])
            max_words = int(expected_length.split('-')[1].split()[0])
        else:
            min_words = marks * 30
            max_words = marks * 70
        
        # Count words
        word_count = len(answer.split())
        
        # Check for key indicators
        has_introduction = any(keyword in answer.lower()[:100] for keyword in ['given', 'define', 'introduction'])
        has_conclusion = any(keyword in answer.lower()[-100:] for keyword in ['therefore', 'hence', 'thus', 'conclusion'])
        has_formula = any(char in answer for char in ['=', '→', '∴', '∵'])
        has_steps = bool(re.search(r'\b(step|first|second|then|next|finally)\b', answer.lower()))
        has_units = bool(re.search(r'\b(cm|m|km|kg|g|s|min|hr|°C|K|Rs|₹)\b', answer))
        
        # Length assessment
        if word_count < min_words:
            length_status = "Too short"
            length_score = 0.5
        elif word_count > max_words * 1.5:
            length_status = "Too long"
            length_score = 0.7
        elif word_count > max_words:
            length_status = "Slightly long"
            length_score = 0.9
        else:
            length_status = "Appropriate"
            length_score = 1.0
        
        # Structure assessment
        structure_score = 0
        if marks >= 3:
            if has_introduction:
                structure_score += 0.3
            if has_conclusion:
                structure_score += 0.3
            if has_steps or has_formula:
                structure_score += 0.4
        else:
            structure_score = 1.0 if word_count > 0 else 0
        
        # Overall quality
        overall_score = (length_score + structure_score) / 2
        
        if overall_score >= 0.8:
            quality = "Good"
        elif overall_score >= 0.6:
            quality = "Satisfactory"
        elif overall_score >= 0.4:
            quality = "Needs Improvement"
        else:
            quality = "Poor"
        
        return {
            "word_count": word_count,
            "expected_range": f"{min_words}-{max_words} words",
            "length_status": length_status,
            "has_introduction": has_introduction,
            "has_conclusion": has_conclusion,
            "has_formula": has_formula,
            "has_steps": has_steps,
            "has_units": has_units,
            "structure_score": round(structure_score, 2),
            "length_score": round(length_score, 2),
            "overall_quality": quality,
            "quick_feedback": cls._generate_quick_feedback(quality, marks, word_count, min_words)
        }
    
    @classmethod
    def _generate_quick_feedback(cls, quality: str, marks: int, word_count: int, min_words: int) -> str:
        """Generate quick feedback based on quality assessment"""
        feedback_templates = {
            "Good": f"Well-structured answer appropriate for {marks} marks. Keep it up!",
            "Satisfactory": f"Decent attempt. Consider adding more detail for {marks} marks.",
            "Needs Improvement": f"Answer needs more content. Aim for at least {min_words} words for {marks} marks.",
            "Poor": f"Answer is too brief. A {marks}-mark question requires more comprehensive response."
        }
        
        return feedback_templates.get(quality, "Please provide a complete answer.")

# Additional helper functions for enhanced evaluation

def get_marking_scheme_summary(board: str = "CBSE") -> Dict[str, str]:
    """Get summary of marking scheme for a board"""
    marking_schemes = {
        "CBSE": {
            "philosophy": "Step marking with emphasis on method",
            "partial_credit": "Yes, for correct approach",
            "alternative_methods": "Full marks if correct",
            "presentation": "5-10% weightage",
            "negative_marking": "No",
                        "grace_marks": "Up to 5% for overall performance"
        },
        "ICSE": {
            "philosophy": "Comprehensive evaluation with depth",
            "partial_credit": "Limited, accuracy emphasized",
            "alternative_methods": "Must be equally rigorous",
            "presentation": "15-20% weightage",
            "negative_marking": "No",
            "grace_marks": "Strict evaluation, minimal grace"
        },
        "SSC": {
            "philosophy": "Liberal marking for basic understanding",
            "partial_credit": "Yes, generous for attempts",
            "alternative_methods": "Accepted if logical",
            "presentation": "5% weightage",
            "negative_marking": "No",
            "grace_marks": "Liberal grace marking"
        }
    }
    
    return marking_schemes.get(board.upper(), marking_schemes["CBSE"])

def get_answer_writing_tips(marks: int, subject: str = "General") -> List[str]:
    """Get answer writing tips based on marks and subject"""
    
    general_tips = {
        1: [
            "Read question carefully - every word matters",
            "Write precise answer - no extra information",
            "For MCQ, mark clearly (darken circle/tick)",
            "Time limit: 1-2 minutes maximum",
            "Review: Check you've answered what's asked"
        ],
        2: [
            "Start with definition or formula",
            "Give one clear example",
            "Use bullet points if multiple parts",
            "Include units in final answer",
            "Time limit: 3-4 minutes"
        ],
        3: [
            "Structure: Introduction → Explanation → Example → Conclusion",
            "Include diagram if it adds value",
            "Show all working for calculations",
            "Underline key terms",
            "Time limit: 5-6 minutes"
        ],
        4: [
            "Read case study completely first",
            "Identify what each part asks",
            "Use data from graphs/tables",
            "Show connections between concepts",
            "Time limit: 7-8 minutes"
        ],
        5: [
            "Plan answer before writing",
            "Cover all aspects comprehensively",
            "Include alternative methods",
            "Draw clear, labeled diagrams",
            "Strong conclusion essential",
            "Time limit: 10-12 minutes"
        ]
    }
    
    subject_specific = {
        "Mathematics": [
            "State formula before using",
            "Show every step clearly",
            "Include rough work",
            "Verify answer if possible",
            "Use proper mathematical notation"
        ],
        "Science": [
            "Use scientific terminology",
            "Include labeled diagrams",
            "Mention SI units always",
            "Add real-life applications",
            "State laws/principles clearly"
        ],
        "Social Science": [
            "Use chronological order for history",
            "Include maps for geography",
            "Provide current examples",
            "Use flowcharts for processes",
            "Link to contemporary issues"
        ],
        "English": [
            "Quote from text if given",
            "Use proper punctuation",
            "Vary sentence structure",
            "Include literary devices",
            "Express personal opinion clearly"
        ]
    }
    
    tips = general_tips.get(marks, [])
    if subject in subject_specific:
        tips.extend(subject_specific[subject])
    
    return tips

def get_common_exam_mistakes() -> Dict[str, List[str]]:
    """Get common exam mistakes by category"""
    
    return {
        "Time Management": [
            "Spending too much time on one question",
            "Not reading all questions first",
            "Missing easy questions at the end",
            "No time left for revision",
            "Poor question selection order"
        ],
        "Answer Writing": [
            "Not reading question completely",
            "Missing command words (explain/describe/list)",
            "Writing irrelevant information",
            "Poor handwriting in hurry",
            "Forgetting units in answers"
        ],
        "Presentation": [
            "No proper margins",
            "Overcrowded answers",
            "Missing question numbers",
            "Diagrams without labels",
            "Final answer not highlighted"
        ],
        "Content": [
            "Using wrong formula",
            "Calculation errors",
            "Missing key points",
            "No examples when asked",
            "Incomplete answers"
        ],
        "Strategy": [
            "Attempting questions in sequence only",
            "Leaving questions blank",
            "Not using rough space",
            "Ignoring mark allocation",
            "Panic during exam"
        ]
    }

def get_last_minute_tips(board: str = "CBSE") -> List[str]:
    """Get last minute exam preparation tips"""
    
    base_tips = [
        "🔍 Review all formulas and important dates",
        "📝 Practice 2-3 previous year papers",
        "⏰ Plan your exam time strategy",
        "💤 Get proper sleep before exam",
        "🎒 Prepare exam kit (pens, pencils, instruments)",
        "📖 Revise chapter summaries only",
        "🎯 Focus on high-weightage topics",
        "✍️ Practice neat handwriting",
        "🧘 Stay calm and confident"
    ]
    
    board_specific_tips = {
        "CBSE": [
            "📚 NCERT examples are most important",
            "🔄 Revise NCERT exercise questions",
            "💡 Remember value-based questions"
        ],
        "ICSE": [
            "📊 Review all reference book examples",
            "🌍 Update current affairs knowledge",
            "🔬 Revise practical-based questions"
        ],
        "SSC": [
            "📍 Focus on state board textbook only",
            "🗺️ Review local geography/history",
            "✏️ Practice diagram drawing"
        ]
    }
    
    tips = base_tips.copy()
    if board.upper() in board_specific_tips:
        tips.extend(board_specific_tips[board.upper()])
    
    return tips

def calculate_question_paper_statistics(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics for a question paper"""
    
    total_marks = sum(q.get('marks', 0) for q in questions)
    total_questions = len(questions)
    
    # Group by marks
    marks_distribution = {}
    for q in questions:
        marks = q.get('marks', 0)
        if marks not in marks_distribution:
            marks_distribution[marks] = 0
        marks_distribution[marks] += 1
    
    # Calculate time needed
    time_needed = 0
    for marks, count in marks_distribution.items():
        time_per_q = {1: 2, 2: 4, 3: 6, 4: 8, 5: 12}.get(marks, marks * 2)
        time_needed += time_per_q * count
    
    # Difficulty distribution (if provided)
    difficulty_dist = {"Easy": 0, "Medium": 0, "Hard": 0}
    for q in questions:
        diff = q.get('difficulty', 'Medium')
        difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1
    
    return {
        "total_marks": total_marks,
        "total_questions": total_questions,
        "marks_distribution": marks_distribution,
        "estimated_time": f"{time_needed} minutes",
        "difficulty_distribution": difficulty_dist,
        "average_marks_per_question": round(total_marks / total_questions, 1) if total_questions > 0 else 0,
        "recommended_time": "180 minutes (3 hours)",
        "time_pressure": "High" if time_needed > 150 else "Moderate" if time_needed > 120 else "Low"
    }

# Validation functions

def validate_answer_format(answer: str, marks: int, subject: str) -> Dict[str, bool]:
    """Validate if answer follows expected format"""
    
    validations = {
        "has_answer": len(answer.strip()) > 0,
        "minimum_length": len(answer.split()) >= marks * 20,
        "maximum_length": len(answer.split()) <= marks * 100,
        "has_structure": marks <= 2 or all(keyword in answer.lower() 
                                          for keyword in ['given', 'therefore']),
        "proper_conclusion": marks <= 2 or any(keyword in answer.lower()[-50:] 
                                              for keyword in ['therefore', 'hence', 'thus'])
    }
    
    # Subject-specific validations
    if subject == "Mathematics":
        validations["has_formula"] = '=' in answer or marks == 1
        validations["has_calculations"] = any(char in answer for char in '+-*/=') or marks == 1
    elif subject == "Science":
        validations["has_units"] = any(unit in answer for unit in ['m', 'kg', 's', '°C']) or marks <= 2
        validations["has_scientific_terms"] = marks == 1 or len(answer) > 50
    
    return validations

# Export all enhanced functionality
__all__ = [
    'CBSEEvaluationMetrics',
    'get_marking_scheme_summary',
    'get_answer_writing_tips',
    'get_common_exam_mistakes',
    'get_last_minute_tips',
    'calculate_question_paper_statistics',
    'validate_answer_format'
]