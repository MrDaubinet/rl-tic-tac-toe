<script lang="ts">
	import { onMount } from 'svelte';
	import Board from "../components/Board.svelte"
	import Score from "../components/Score.svelte"
	import Turn from "../components/Turn.svelte"
	import History from "../components/History.svelte"
	import Notification from "../components/Notification.svelte"
	import { gameStore } from '../lib/GameController';
	import SettingsButton from '../components/SettingsButton.svelte';
	import Settings from '../components/Settings.svelte';
	import { derived } from 'svelte/store';
	import { get } from 'svelte/store';

	// Destructure the stores
	const {
		agentType,
		agent_play,
		agentPlayer,
		showHistory,
		show_notification,
		notification,
		turn,
		state,
		score,
		agentLoading,
		showValueFunction,
		getValueForCell,
	} = gameStore;
	const availableAgentTypes = gameStore.availableAgentTypes;

	let showSettings = false;

	// Compute the value for each cell reactively
	const cellValues = derived(
		[state, turn, agentLoading, showValueFunction],
		([$state, $turn, $agentLoading, $showValueFunction]) => {
			if ($agentLoading || !$showValueFunction) return Array(9).fill(null);
			return $state.map((v, i) => getValueForCell(i));
		}
	);

	onMount(() => {
		gameStore.updateState();
		gameStore.loadAgent();
	});

	function handleShowSettings(val: boolean) {
		showSettings = val;
		if (val) showHistory.set(false);
	}

	function handleShowHistory(val: boolean) {
		showHistory.set(val);
		if (val) showSettings = false;
	}

	function toggleShowValueFunction(val: boolean) {
		showValueFunction.set(val);
	}

	// Reactive statement to call loadAgent when agentType or agentPlayer changes
	$: if ($agentType && $agentPlayer) {
		gameStore.loadAgent();
	}
</script>

<!-- Notification -->
{#if $show_notification}
	<Notification
	message={$notification}
	onClose={() => show_notification.set(false)}
	/>
{/if}

<div class="fixed top-2 right-2 z-50">
	<SettingsButton onClick={() => handleShowSettings(!showSettings)} />
</div>

<div class="max-w-4xl mx-auto px-4 py-8">
	<div class="text-center mb-8">
		<h1 class="text-2xl md:text-4xl font-bold mb-2">Tic Tac Toe</h1>
		<p class="text-gray-400 text-sm md:text-base">Play against AI agents trained with reinforcement learning</p>
	</div>

	<!-- Responsive grid: 1 col, 2 col, or 3 col depending on settings/history -->
	<div class="grid grid-cols-1 {(showSettings || $showHistory) ? 'md:grid-cols-2' : ''}">
		<div class="mt-4 mb-8">
			<div class="text-center">
				<Turn turn={$turn} agent_play={$agent_play} />
			</div>
			
			<div class="flex justify-center mt-4">
				<Board
					state={$state}
					selectMarker={(i: number) => gameStore.selectMarker(i)}
					cellValues={$cellValues}
				/>
			</div>
			<div class="text-center mt-8">
				<Score score={$score} agent_play={$agent_play} />
			</div>
			<div class="flex justify-center mt-6 gap-4">
				<button
					class="px-6 py-2 text-3xl text-white font-semibold transition hover:text-gray-300 hover:animate-pulse"
					on:click={() => gameStore.reset(0)}
				>
					Reset Game
				</button>
				<button
					type="button"
					class="px-6 py-2 text-3xl text-white font-semibold transition hover:text-gray-300 hover:animate-pulse"
					aria-pressed={$showHistory}
					on:click={() => handleShowHistory(!$showHistory)}
				>
					{$showHistory ? 'Hide History' : 'Show History'}
				</button>
			</div>
		</div>
		
		{#if showSettings}
			<div class="px-4 lg:px-10">
				<Settings
					agentType={$agentType}
					agentPlayer={$agentPlayer}
					availableAgentTypes={availableAgentTypes}
					onAgentTypeChange={(type) => agentType.set(type as typeof $agentType)}
					onAgentPlayerChange={(player) => {
						agentPlayer.set(player);
						// Wait for the store to update, then check if agent should go first
						setTimeout(() => {
						if (player === 'X' && $agent_play) {
							gameStore.makeAgentMove();
						}
						}, 0);
					}}
					showValueFunction={$showValueFunction}
					onShowValueFunctionChange={toggleShowValueFunction}
				/>
			</div>
		{/if}

		{#if $showHistory}
			<div class="px-4 lg:px-10">
				<History updateState={() => {
					gameStore.updateState();
					// After updating state, check if it's the agent's turn and trigger agent move
					const agent_play_val = get(agent_play);
					const turn_val = get(turn);
					const agentPlayer_val = get(agentPlayer);
					if (agent_play_val && ((agentPlayer_val === 'X' && turn_val) || (agentPlayer_val === 'O' && !turn_val))) {
						gameStore.makeAgentMove();
					}
				}} />
			</div>
		{/if}
	</div>
</div> 
